# coding = utf-8

""" define the U-Net structure """

import torch
from torch import nn
from .utils import _initialize_weights
import torch.nn.functional as F


class WNet(nn.Module):
    """ define W-Net for help segmentation with boundary detection first """
    def __init__(self, in_channel, inter_channel, out_channel, bound_net, pre_train_bound_path,
                 seg_net='res_unet_dp', pretrain=False):
        super(WNet, self).__init__()

        # set plaque segmentation network structure
        if bound_net == 'unet':
            from .unet import UNet28 as UNet
            self.BoundNet = UNet(in_channel, inter_channel)

        elif bound_net == 'res_unet':
            from .res_unet import ResUNet28 as ResUNet
            self.BoundNet = ResUNet(in_channel, inter_channel)

        elif bound_net == 'res_unet_dp':
            from .res_unet_dp import ResUNet28 as ResUNet
            self.BoundNet = ResUNet(in_channel, inter_channel, p=0.0)

        elif bound_net == 'tiramisu':
            from .tiramisu import FCDenseNet67 as FCDenseNet
            self.BoundNet = FCDenseNet(in_channel, inter_channel, theta=1.0)

        elif bound_net == 'hyper_tiramisu':
            from .hyper_tiramisu import FCDenseNet67 as FCDenseNet
            self.BoundNet = FCDenseNet(in_channel, inter_channel, theta=1.0)

        elif bound_net == 'deeplab_resnet':
            from .deeplab_resnet import Res_Ms_Deeplab
            self.BoundNet = Res_Ms_Deeplab(in_channel, inter_channel)
        self.BoundNet.load_state_dict(torch.load("{}/model.pth".format(pre_train_bound_path),
                                                 map_location=lambda storage, loc: storage))


        self.BoundNet = torch.load("{}/model.pth".format(pre_train_bound_path),
                           map_location=lambda storage, loc: storage)

        # set plaque segmentation network structure
        if pretrain:
            pass # define the pre_train_seg_path here
        else:
            if seg_net == 'unet':
                from .unet import UNet28 as UNet
                self.SegNet = UNet(inter_channel, out_channel)

            elif seg_net == 'res_unet':
                from .res_unet import ResUNet28 as ResUNet
                self.SegNet = ResUNet(inter_channel, out_channel)

            elif seg_net == 'res_unet_dp':
                from .res_unet_dp import ResUNet28 as ResUNet
                self.SegNet = ResUNet(inter_channel, out_channel, p=0.0)

            elif seg_net == 'tiramisu':
                from .tiramisu import FCDenseNet67 as FCDenseNet
                self.SegNet = FCDenseNet(inter_channel, out_channel, theta=1.0)

            elif seg_net == 'hyper_tiramisu':
                from .hyper_tiramisu import FCDenseNet67 as FCDenseNet
                self.SegNet = FCDenseNet(inter_channel, out_channel, theta=1.0)

            elif seg_net == 'deeplab_resnet':
                from .deeplab_resnet import Res_Ms_Deeplab
                self.SegNet = Res_Ms_Deeplab(inter_channel, out_channel)

    def forward(self, x):
        # size of bound_pred: [B, C_out, H, W] size of x: [B, C_in, H, W]
        bound_pred = self.BoundNet(x)  # for innerouter bound,  out_channel=3 else 2

        x_bound = torch.cat([x, F.softmax(bound_pred, dim=1)[:, 1:]], dim=1)
        y = self.SegNet(x_bound)

        return bound_pred, y

def WNetPT(in_channel, inter_channel, out_channel, bound_net, pre_train_bound_path, seg_net='res_unet_dp'):
    model = WNet(in_channel, inter_channel, out_channel, bound_net, pre_train_bound_path,
                 seg_net, pretrain=False)

    return model