# _*_ coding: utf-8 _*_

from torch import nn

def count_parameters(model):
    """ count number of parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _initialize_weights(model):
    """ model weight initialization """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()