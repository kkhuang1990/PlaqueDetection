# _*_ coding: utf-8 _*_

""" define custom learning rate scheduler """

from __future__ import print_function

from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """ poly learning rate scheduler """
    def __init__(self, optimizer, max_iter=100, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]