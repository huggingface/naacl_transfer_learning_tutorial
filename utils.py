# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import logging
import os
from tqdm import tqdm
from pprint import pformat

import torch

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler, OutputHandler, TensorboardLogger

from pytorch_pretrained_bert import cached_path, BertTokenizer

DATASETS_URL = {
    'wikitext-2':   {'train': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-2/train.txt",
                     'valid': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-2/valid.txt"},
    'wikitext-103': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.train.tokens",
                     'valid': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.valid.tokens"},
    'simplebooks-2-raw': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-2-raw/train.txt",
                          'valid': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-2-raw/valid.txt"},
    'simplebooks-92-raw': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-92-raw/train.txt",
                           'valid': "https://s3.amazonaws.com/datasets.huggingface.co/simplebooks-92-raw/valid.txt"},
    'imdb': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/train.txt",
             'valid': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/valid.txt"},
    'trec': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/trec/train.txt",
             'valid': "https://s3.amazonaws.com/datasets.huggingface.co/trec/test.txt"},
    }

DATASETS_LABELS_URL = {
    'imdb': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/train.labels.txt",
             'valid': "https://s3.amazonaws.com/datasets.huggingface.co/aclImdb/valid.labels.txt"},
    'trec': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/trec/train.labels.txt",
             'valid': "https://s3.amazonaws.com/datasets.huggingface.co/trec/test.labels.txt"},
    }

DATASETS_LABELS_CONVERSION = {
    'imdb':         {'pos': 0, 'neg': 1},
    'trec':         {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5},
    }

PRETRAINED_MODEL_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/"

WEIGHTS_NAME = 'model_checkpoint.pth'
CONFIG_NAME = 'model_training_args.bin'

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0, to_left=True):
    """ Pad a dataset (list of list) to the left or the right. """
    max_l = max(len(x) for x in dataset)
    dataset = [(x if to_left else []) + [padding] * (max_l - len(x)) + ([] if to_left else x) for x in dataset]
    return dataset


def add_logging_and_checkpoint_saving(trainer, evaluator, metrics, model, optimizer, args, prefix=""):
    """ Add to training engine tensorboard logging, progress bar with average loss, checkpoint saving and save training config. """
    # Add progress bar with average loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, prefix + "loss")
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=[prefix + "loss"])
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

    # Add tensorboard logging with training and evaluation metrics
    tb_logger = TensorboardLogger(log_dir=None)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=[prefix + "loss"]),
                              event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer),
                              event_name=Events.ITERATION_STARTED)
    @evaluator.on(Events.COMPLETED)
    def tb_log_metrics(engine):
        for name in metrics.keys():
            tb_logger.writer.add_scalar(name, engine.state.metrics[name], trainer.state.iteration)

    # Add checkpoint saving after each epoch - take care of distributed encapsulation ('getattr()')
    checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})

    # Save training configuration
    torch.save(args, os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))

    return checkpoint_handler, tb_logger


def get_and_tokenize_dataset(tokenizer, dataset_dir='wikitext-103', dataset_cache=None, with_labels=False):
    """ Retrieve, tokenize, encode and cache a dataset with optional labels """
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load encoded dataset from cache at %s", dataset_cache)
        encoded_dataset = torch.load(dataset_cache)
    else:
        # If the dataset is in our list of DATASETS_URL, use this url, otherwise, look for 'train.txt' and 'valid.txt' files
        if dataset_dir in DATASETS_URL:
            dataset_map = DATASETS_URL[dataset_dir]
        else:
            dataset_map = {'train': os.path.join(dataset_dir, 'train.txt'),
                           'valid': os.path.join(dataset_dir, 'valid.txt')}

        logger.info("Get dataset from %s", dataset_dir)
        # Download and read dataset and replace a few token for compatibility with the Bert tokenizer we are using
        dataset = {}
        for split_name in dataset_map.keys():
            dataset_file = cached_path(dataset_map[split_name])
            with open(dataset_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                dataset[split_name] = [
                        line.strip(' ').replace('<unk>', '[UNK]').replace('\n', '[SEP]' if not with_labels else '')
                        for line in tqdm(all_lines)]

        # If we have labels, download and and convert labels in integers
        labels = {}
        if with_labels:
            label_conversion_map = DATASETS_LABELS_CONVERSION[dataset_dir]
            for split_name in DATASETS_LABELS_URL[dataset_dir]:
                dataset_file = cached_path(DATASETS_LABELS_URL[dataset_dir][split_name])
                with open(dataset_file, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                    labels[split_name] = [label_conversion_map[line.strip()] for line in tqdm(all_lines)]

        # Tokenize and encode the dataset
        logger.info("Tokenize and encode the dataset")
        logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)  # No warning on sample size
        def encode(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, encode(o)) for n, o in obj.items())
            return list(encode(o) for o in tqdm(obj))
        encoded_dataset = encode(dataset)

        # Add labels if needed, or if we are doing language modeling, add number of words to get word-level ppl and gather in one list
        for split_name in ['train', 'valid']:
            if with_labels:
                encoded_dataset[split_name + '_labels'] = labels[split_name]
            else:
                encoded_dataset[split_name] = [ind for line in encoded_dataset[split_name] for ind in line]
                encoded_dataset[split_name + '_num_words'] = sum(len(line.split(' ')) for line in dataset[split_name])

        # Save to cache
        if dataset_cache:
            logger.info("Save encoded dataset to cache at %s", dataset_cache)
            torch.save(encoded_dataset, dataset_cache)

    return encoded_dataset


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    get_and_tokenize_dataset(tokenizer, dataset_dir='imdb', dataset_cache=None, with_labels=True)


