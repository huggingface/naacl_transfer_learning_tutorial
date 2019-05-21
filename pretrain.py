# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.
import logging
import math
import random
import os
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pprint import pformat

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch import nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from ignite.contrib.handlers import PiecewiseLinear, ProgressBar, LRScheduler, ConcatScheduler, CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import (OptimizerParamsHandler,
                                                        OutputHandler,
                                                        TensorboardLogger)
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage

from pytorch_pretrained_bert import BertTokenizer

from model import TransformerWithLMHead
from utils import get_and_tokenize_dataset, average_distributed_scalar

logger = logging.getLogger(__file__)

WEIGHTS_NAME = 'model_checkpoint.pth'
CONFIG_NAME = 'model_training_args.bin'

# def randomize_dataset_blocks(args, dataloader):
#     """ Add some diversity in the dataset at each epoch to reduce overfitting """
#     shift = random.randrange(1, args.num_max_positions)
#     seq_length = random.choice((int(args.num_max_positions / 2), args.num_max_positions))
#     dataset = dataloader.dataset.view(-1)
#     out_dataset = torch.empty_like(dataset)
#     out_dataset[shift:] = dataset[:-shift]
#     out_dataset[:shift] = dataset[-shift:]
#     dataloader.dataset = out_dataset.view(-1, seq_length)

def get_data_loaders(args, tokenizer):
    """ Prepare the dataloaders for training and evaluation """
    datasets = get_and_tokenize_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Convert to Tensor and reshape")
    for split_name in ['train', 'valid']:
        tensor = torch.tensor(datasets[split_name], dtype=torch.long)
        num_sequences = (tensor.size(0) // args.num_max_positions) * args.num_max_positions
        datasets[split_name] = tensor.narrow(0, 0, num_sequences).view(-1, args.num_max_positions)

    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(datasets['train']) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(datasets['valid']) if args.distributed else None
    train_loader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(datasets['valid'], sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(datasets['train'].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(datasets['valid'].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler, datasets['train_num_words'], datasets['valid_num_words']


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='wikitext-2', help="One of ('wikitext-103', 'wikitext-2') or a dict of splits paths.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")

    parser.add_argument("--embed_dim", type=int, default=410, help="Embeddings dim")
    parser.add_argument("--hidden_dim", type=int, default=2100, help="Hidden dimension")
    parser.add_argument("--num_max_positions", type=int, default=256, help="Max input length")
    parser.add_argument("--num_heads", type=int, default=10, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=16, help="NUmber of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Dropout")
    parser.add_argument("--sinusoidal_embeddings", action="store_true", help="Use sinusoidal embeddings instead of learned ones")

    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    # parser.add_argument("--lr_gamma", type=float, default=0.9999, help="Learning rate exponential decrease coefficient")
    parser.add_argument("--max_norm", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n_warmup", type=float, default=0.001, help="Ratio of warmup iterations to total training")
    parser.add_argument("--eval_every", type=int, default=-1, help="Evaluate every X steps (-1 => end of epoch)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradient")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
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
    args.num_embeddings = len(tokenizer.vocab)  # We need this to create the model at next line (number of embeddings to use)
    model = TransformerWithLMHead(args)
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info("Model has %s parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Prepare model for distributed training if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler, train_num_words, valid_num_words = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = batch.transpose(0, 1).contiguous().to(args.device)  # to shape [seq length, batch]
        logits, loss = model(batch, labels=batch)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = batch.transpose(0, 1).contiguous().to(args.device)  # to shape [seq length, batch]
            logits = model(batch)
            shift_logits = logits[:-1].view(-1, logits.size(-1))
            shift_labels = batch[1:].view(-1)
            return shift_logits, shift_labels
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate at the end of each epoch and every 'eval_every' iterations if needed
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_every > 0:
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                lambda engine: evaluator.run(val_loader) if engine.state.iteration % args.eval_every == 0 else None)
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

    # Randomize a bit on each epoch
    # trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: randomize_dataset_blocks(args, train_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Learning rate schedule: linearly warm-up to lr and then decrease the learning rate to zero
    cos_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, len(train_loader) / args.train_batch_size * args.n_epochs)
    # exp_scheduler = LRScheduler(lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=args.lr_gamma))
    scheduler = create_lr_scheduler_with_warmup(cos_scheduler, warmup_start_value=0.0, warmup_end_value=args.lr,
                    warmup_duration=int(len(train_loader) / args.train_batch_size * args.n_epochs * args.n_warmup))
    # scheduler = PiecewiseLinear(optimizer, "lr", [(0, 0.0), (int(args.n_epochs * len(train_loader) * args.n_warmup), args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we average distributed metrics using average_distributed_scalar
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    metrics["average_word_ppl"] = MetricsLambda(lambda x: math.exp(x * val_loader.dataset.numel() / valid_num_words), metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model and configuration before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)

        @evaluator.on(Events.COMPLETED)  # Log evaluator metrics on tensorboard
        def tb_log_metrics(engine):
            for name in metrics.keys():
                tb_logger.writer.add_scalar(name, engine.state.metrics[name], trainer.state.iteration)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint for easy re-loading
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
