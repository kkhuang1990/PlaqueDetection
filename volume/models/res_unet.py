# coding = utf-8

""" define the U-Net structure """

import torch
from torch import nn
from .utils import _initialize_weights

def conv_333(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class ResBlock(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = conv_333(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv_333(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual

        return out

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

class ResUNet(nn.Module):
    """ UNet class """

    def __init__(self, in_channels=1, out_channels=5, down_blocks=[32, 64, 128, 256, 512],
                 up_blocks = [512, 256, 128, 64, 32], bottleneck = 1024, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock(input_channel, output_channel, stride=2, p=p))

        # bottleneck block
        self.bottleneck  = ResBlock(self.down_blocks[-1], bottleneck, stride=2, p=p)

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl = nn.Conv3d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights(self)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        skip_connections = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            skip_connections.append(out)
            # print(out.size())

        out = self.bottleneck(out)
        # print(out.size())

        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.TransUpBlocks[b_inx](skip, out)
            out = self.BlocksUp[b_inx](out)
            # print(out.size())

        output = self.fl(out)
        # print(out.size())

        return output

def ResUNet28(in_channels, out_channels, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, down_blocks=[32, 64, 128, 256, 512],
                 up_blocks = [512, 256, 128, 64, 32], bottleneck = 1024, p=p)


def ResUNet23(in_channels, out_channels, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=p)

def ResUNet18(in_channels, out_channels, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, down_blocks=[32, 64, 128],
                 up_blocks = [128, 64, 32], bottleneck = 256, p=p)

if __name__ == "__main__":
    in_channels = 1
    out_channels = 2
    unet = ResUNet28(in_channels, out_channels, p=0.0)
    print(unet)
    x = torch.FloatTensor(6, 1, 32, 96, 96)
    y = unet(x)