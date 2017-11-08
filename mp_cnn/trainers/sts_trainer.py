import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mp_cnn.trainers.general_multiclass_trainer import GeneralMulticlassTrainer


class STSTrainer(GeneralMulticlassTrainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(STSTrainer, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
