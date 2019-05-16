import os
import logging
import tarfile

from pytorch_pretrained_bert import cached_path

WIKITEXT_2_URL = "https://s3.amazonaws.com/datasets.huggingface.co/wikitext-2.tar.gz"

logger = logging.getLogger(__file__)

def download_dataset(dataset_dir, dataset_url=WIKITEXT_2_URL):
    """ Download and extract wikitext-2 from S3 """
    resolved_archive_file = cached_path(dataset_url)

    logger.info("extracting dataset file {} to data dir {}".format(resolved_archive_file, dataset_dir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(dataset_dir)
    return dataset_dir

def get_dataset(tokenizer, dataset_dir, dataset_cache=None):
    """ Retrieve, tokenize, encode and cache the dataset """
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_dir)
        dataset_dir = download_dataset(dataset_dir)
        dataset = {}
        for split_name in ['test', 'train', 'valid']:
            with open(os.path.join(dataset_dir, split_name + '.txt'), "r", encoding="utf-8") as f:
                dataset[split_name] = f.readlines()

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.encode(obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset
