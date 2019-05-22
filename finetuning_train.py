# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.
import logging
import importlib
import os
from argparse import ArgumentParser
from pprint import pformat

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ignite.contrib.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, MetricsLambda

from pytorch_pretrained_bert import BertTokenizer, cached_path

from utils import (get_and_tokenize_dataset, average_distributed_scalar, pad_dataset,
                   add_logging_and_checkpoint_saving, WEIGHTS_NAME, CONFIG_NAME, PRETRAINED_MODEL_URL)

logger = logging.getLogger(__file__)

def get_data_loaders(args, tokenizer, trim_length, add_clf_token=None):
    """ Prepare the dataloaders for training and evaluation.
        Add a classification token at the end of each sample if needed. """
    datasets = get_and_tokenize_dataset(tokenizer, args.dataset_path, args.dataset_cache, with_labels=True)

    logger.info("Convert to Tensor, pad and trim to trim_length")
    tensor_datasets = {}
    for split_name in ['train', 'valid']:
        dataset = pad_dataset(datasets[split_name])
        tensor = torch.tensor(dataset, dtype=torch.long)
        tensor = tensor.narrow(-1, 0, trim_length)
        if add_clf_token is not None:
            tensor = torch.cat([tensor[:, :-1], torch.full((len(tensor), 1), add_clf_token, dtype=torch.long)], dim=-1)
        labels = torch.tensor(datasets[split_name + '_labels'], dtype=torch.long)
        tensor_datasets[split_name] = (tensor, labels)

    logger.info("Build train and validation dataloaders")
    train_dataset = TensorDataset(*tensor_datasets['train'])
    valid_dataset = TensorDataset(*tensor_datasets['valid'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(datasets['train']) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(datasets['valid']) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default=PRETRAINED_MODEL_URL, help="Path to the pretrained model checkpoint")
    parser.add_argument("--dataset_path", type=str, default='imdb', help="'imdb' or a dict of splits paths.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache_fine_tune', help="Path or url of the dataset cache")

    parser.add_argument("--finetuning_model_class", type=str, default="TransformerWithClfHead", help="Fine-tuning model class for the target task")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for the target classification task")
    parser.add_argument("--adapters_dim", type=int, default=-1, help="If >0 add adapters to the model wtih adapters_dim dimension")

    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--n_warmup", type=float, default=0, help="Number of warmup iterations")
    parser.add_argument("--eval_every", type=int, default=-1, help="Evaluate every X steps (-1 => end of epoch)")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log on main process only, logger.warning => log on all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))  # This is a logger.info: only printed on the first process

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, model and optimizer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)  # Let's use a pre-defined tokenizer

    logger.info("Create model from class %s and configuration %s", args.finetuning_model_class, os.path.join(args.model_checkpoint, CONFIG_NAME))
    ModelClass = getattr(importlib.import_module("finetuning_model"), args.finetuning_model_class)
    pretraining_args = torch.load(cached_path(os.path.join(args.model_checkpoint, CONFIG_NAME)))
    model = ModelClass(config=pretraining_args, fine_tuning_config=args).to(args.device)

    logger.info("Load pretrained weigths from %s", os.path.join(args.model_checkpoint, WEIGHTS_NAME))
    state_dict = torch.load(cached_path(os.path.join(args.model_checkpoint, WEIGHTS_NAME)), map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info("Model has %s parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Prepare model for distributed training if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    loaders = get_data_loaders(args, tokenizer, pretraining_args.num_max_positions, add_clf_token=tokenizer.vocab['[CLS]'])
    train_loader, val_loader, train_sampler, valid_sampler = loaders

    # Training function and trainer
    def update(engine, batch):
        model.train()
        inputs, labels = (t.to(args.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous()  # to shape [seq length, batch]
        logits, losses = model(inputs, clf_labels=labels)
        loss = sum(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, labels = (t.to(args.device) for t in batch)
            inputs = inputs.transpose(0, 1).contiguous()  # to shape [seq length, batch]
            lm_logits, clf_logits = model(inputs)
            return clf_logits, labels
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate at the end of each epoch and every 'eval_every' iterations if needed
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_every > 0:
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                lambda engine: evaluator.run(val_loader) if engine.state.iteration % args.eval_every == 0 else None)
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Learning rate schedule: linearly warm-up to lr and then decrease the learning rate to zero with cosine schedule
    cos_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, len(train_loader) * args.n_epochs)
    scheduler = create_lr_scheduler_with_warmup(cos_scheduler, 0.0, args.lr, args.n_warmup)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we average distributed metrics using average_distributed_scalar
    metrics = {"accuracy": Accuracy()}
    metrics.update({"average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model and configuration before we start to train
    if args.local_rank in [-1, 0]:
        checkpoint_handler = add_logging_and_checkpoint_saving(trainer, evaluator, metrics, model, optimizer, args)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint for easy re-loading
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
