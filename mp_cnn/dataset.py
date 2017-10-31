from collections import defaultdict
from enum import Enum
import math
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data

from datasets.sick import SICK
from datasets.msrvid import MSRVID

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


unknown_vec = None


def unk(tensor):
    global unknown_vec
    if unknown_vec is None:
        unknown_vec = torch.Tensor(tensor.size())
        unknown_vec.normal_(0, 0.01)
    return unknown_vec


class MPCNNDatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device):
        if dataset_name == 'sick':
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'sick/')
            train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=unk)
            embedding_dim = SICK.TEXT_FIELD.vocab.vectors.size()
            embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
            embedding.weight = nn.Parameter(SICK.TEXT_FIELD.vocab.vectors)
            return SICK, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'msrvid':
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'msrvid/')
            dev_loader = None
            train_loader, test_loader = MSRVID.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=unk)
            embedding_dim = MSRVID.TEXT_FIELD.vocab.vectors.size()
            embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
            embedding.weight = nn.Parameter(MSRVID.TEXT_FIELD.vocab.vectors)
            return MSRVID, embedding, train_loader, test_loader, dev_loader
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

