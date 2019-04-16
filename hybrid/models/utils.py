# _*_ coding: utf-8 _*_

from torch import nn
import math
import torch
torch.manual_seed(42) # make random weight fixed for every running

def count_parameters(model):
    """ count number of parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _initialize_weights_3d(model):
    """ model weight initialization """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def _initialize_weights_2d(model):
    """ model weight initialization """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
