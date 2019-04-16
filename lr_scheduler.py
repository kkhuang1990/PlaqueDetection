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

# def debug_polylr():
#     from torch import optim
#     from image.models.unet import UNet18 as UNet
#     model = UNet(1, 5)
#     # print(model)
#     opt = optim.SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=0.0)
#     lr = PolyLR(optimizer=opt, max_iter=500)
#     print(lr.max_iter)
#
#     for epoch in range(500):
#         lr.step()
#         print(lr.get_lr()[0])
#         opt.zero_grad()
#         opt.step()
#
# if __name__ == "__main__":
#     debug_polylr()