# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import logging
import os
import tarfile
from tqdm import tqdm

import torch

from pytorch_pretrained_bert import cached_path

DATASETS_URL = {
    'wikitext-2':   {'train': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-2/train.txt",
                     'valid': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-2/valid.txt"},
    'wikitext-103': {'train': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.train.tokens",
                     'valid': "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.valid.tokens"}
    }

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_and_tokenize_dataset(tokenizer, dataset_dir='wikitext-103', dataset_cache=None):
    """ Retrieve, tokenize, encode and cache the dataset """
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load encoded dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        if dataset_dir in DATASETS_URL:
            dataset_dir = DATASETS_URL[dataset_dir]
        logger.info("Download dataset from %s", dataset_dir)
        dataset = {}
        for split_name, dataset_path in dataset_dir.items():
            dataset_file = cached_path(full_dataset_path)
            with open(dataset_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                dataset[split_name] = [line.strip() for line in all_lines if len(line.strip())]

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.encode(obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in tqdm(obj))
        dataset = tokenize(dataset)

        # Gather each split in one big list
        for name, data in dataset.items():
            dataset[name] = [ind for line in data for ind in line]

        if dataset_cache:
            logger.info("Save encoded dataset to cache at %s", dataset_cache)
            torch.save(dataset, dataset_cache)

    return dataset
