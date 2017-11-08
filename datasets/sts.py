import math
import os

import numpy as np
import torch
from torchtext.data.field import Field
from torchtext.data.iterator import BucketIterator
from torchtext.data.pipeline import Pipeline
from torchtext.data.dataset import TabularDataset
from torchtext.vocab import Vectors


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(STS.NUM_CLASSES)
    ceil, floor = math.ceil(sim), math.floor(sim)
    if ceil == floor:
        class_probs[floor] = 1
    else:
        class_probs[floor] = ceil - sim
        class_probs[ceil] = sim - floor

    return class_probs


class STS(TabularDataset):
    NAME = 'sts'
    NUM_CLASSES = 6
    TEXT_FIELD = Field(batch_first=True)
    LABEL_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, postprocessing=Pipeline(get_class_probs))

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a STS dataset instance
        """
        fields = [('label', self.LABEL_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD)]
        super(STS, self).__init__(path, 'TSV', fields)

    @classmethod
    def splits(cls, path, train, validation, test, **kwargs):
        return super(STS, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, year, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param year: year of STS competition
        :param vectors_name: name of word vectors file
        :param vectors_cache: directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        if year != 2015:
            raise NotImplementedError('Only STS 2015 is currently supported.')

        train, val, test = cls.splits(path, '2015.train.tsv', '2015.val.tsv', '2015.test.tsv')

        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
