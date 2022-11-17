import os
import time
import numpy as np
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.scheduler_ramps import *
from utils.helpers import *
from .base import *

class Train_Base():
    """
    Class with base train methods.
    """

    def __init__(self, args):
        """
        Initializes the class. Assign all parameters including the model, dataloaders, samplers and extra variables for each method.

        Args:
            args (dictionary): all user defined parameters with some pre-initialized objects (e.g., model, optimizer, dataloaders)
        """
        self.args = args

    def apply_train_common(self, model, optimizer, input_var, target_var, loader_index, len_trainloader, epoch, meters, end):
        """
        Common train set of operations shared between all training methods
        """
        raise NotImplementedError("Normal training unavailable")

    def adjust_learning_rate(self, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
        """
        Adjust learning rate based on lr_rampup and lr_rampdown_epochs parameters pre-defined in self.args.

        Args:
            optimizer: predefined optimizer assigned to model
            epoch: current training epoch
            step_in_epoch: current step in epoch
            total_steps_in_epoch: total steps in epoch

        Returns:
            float: new learning rate
        """
        lr = self.args.lr
        epoch = epoch + step_in_epoch / total_steps_in_epoch

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = linear_rampup(epoch, self.args.lr_rampup) * (self.args.lr - self.args.initial_lr) + self.args.initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.args.lr_rampdown_epochs:
            assert self.args.lr_rampdown_epochs >= self.args.epochs
            lr *= cosine_rampdown(epoch, self.args.lr_rampdown_epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def validate(self, eval_loader, model, epoch, testing = False, k=5):
        """
        Returns current model top-1 and top-k accuracy evaluated on a data loader
        """
        raise NotImplementedError("Normal validation unavailable") 

    def evaluate_after_train(self, modelName, validloader, testloader, model, optimizer, epoch):
        """
        Evaluate and save weights if current validation accuracy is better than previous epoch.
        Log results on console and TensorBoard logger.
        """
        raise NotImplementedError("Normal validation unavailable")

    def save_checkpoint(self, state, is_best, dirpath, epoch, mode='I'):
        """
        Save model weights - checkpoint model

        Args:
            state: current state
            is_best: if the model is best after validation
            dirpath: path to save weights
            epoch: current epoch
            modelName: name to save
        """
        filename = '{}.checkpoint_{}.ckpt'.format(epoch, mode)
        checkpoint_path = os.path.join(dirpath, filename)
        torch.save(state, checkpoint_path)

    def save_best_checkpoint(self, state, is_best, dirpath, epoch, modelName):
        """
        Save model weights - checkpoint current best model state

        Args:
            state: current state
            is_best: if the model is best after validation
            dirpath: path to save weights
            epoch: current epoch
            modelName: name to save
        """
        best_path = os.path.join(dirpath, '{}.best.ckpt'.format(modelName))
        torch.save(state, best_path)

    def accuracy(self, output, target, topk=(1,)):
        """
        Computes the precision-k for the specified values of k
        """
        maxk = max(topk)
        #labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).type(torch.cuda.FloatTensor)
        minibatch_size = len(target)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / minibatch_size))
        return res     