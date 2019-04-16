# _*_ coding: utf-8 _*_

""" 3D U-Net for semantic segmentation """
import torch
from torch import nn
from .utils import _initialize_weights

class ConvBlock(nn.Sequential):
    """ Convolution Block """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv2', nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1))
        self.add_module('relu2', nn.ReLU(True))

    def forward(self, x):
        return super().forward(x)

class UpConv(nn.Module):
    """ up convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2,
                                            stride=2, padding=0)

    def forward(self, skip, x):
        out = self.transconv(x)
        out = torch.cat([skip, out], 1)

        return out

class UNet(nn.Module):
    """ UNet class """

    def __init__(self, in_channels=1, out_channels=5, down_blocks=[32, 64, 128, 256, 512],
                 up_blocks = [512, 256, 128, 64, 32], bottleneck = 1024):
        super().__init__()

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            input_channel = in_channels if b_inx == 0 else self.down_blocks[b_inx-1]
            output_channel = self.down_blocks[b_inx]
            self.BlocksDown.append(ConvBlock(input_channel, output_channel))

        # bottleneck block
        self.bottleneck  = ConvBlock(self.down_blocks[-1], bottleneck)

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ConvBlock(input_channel, output_channel))

        # final convolution layer
        self.fl = nn.Conv3d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights(self)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = x
        # print(out.size())
        skip_connections = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            skip_connections.append(out)
            out = self.maxpool(out)
            # print(out.size())

        out = self.bottleneck(out)
        # print(out.size())

        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.TransUpBlocks[b_inx](skip, out)
            out = self.BlocksUp[b_inx](out)
            # print(out.size())

        output = self.fl(out)
        return output

def UNet28(in_channels, out_channels):
    return UNet(in_channels=in_channels, out_channels=out_channels, down_blocks=[32, 64, 128, 256, 512],
                 up_blocks = [512, 256, 128, 64, 32], bottleneck = 1024)

def UNet23(in_channels, out_channels):
    return UNet(in_channels=in_channels, out_channels=out_channels, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512)

def UNet18(in_channels, out_channels):
    return UNet(in_channels=in_channels, out_channels=out_channels, down_blocks=[32, 64, 128],
                 up_blocks = [128, 64, 32], bottleneck = 256)

# if __name__ == "__main__":
#     in_channels = 1
#     out_channels = 5
#     unet = UNet28(in_channels, out_channels)
#     print(unet)
#     x = torch.FloatTensor(6, 1, 32, 96, 96)
#     y = unet(x)