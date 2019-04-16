# coding = utf-8

""" Hybrid Res-UNet architecture with regularization of number of predicted boundary pixels
    the contract path is 3D while the expansion path is 2D.
    For input, slices before and after current slice are concatenated as a volume.
    For output, annotation of current slice is compared with the prediction (single slice)
"""

import torch
from torch import nn
from .utils import _initialize_weights_2d, _initialize_weights_3d
import torch.nn.functional as F

# 3D convolution
def conv_333(in_channels, out_channels, stride=1, padding=1):
    # here only the X and Y directions are padded and no padding along Z direction
    # in this way, we can make sure the central slice of the input volume will remain central
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=True)

class ResBlock3D(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        padding = 1 if stride == 1 else (0, 1, 1)
        self.conv1 = conv_333(in_channels, out_channels, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv_333(out_channels, out_channels, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=3, stride=stride, bias=False, padding=padding),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            # print("output residual size: {}".format(residual.size()))
        # print("output size: {}".format(out.size()))
        out += residual

        return out

# 2D convolution
def conv_33(in_channels, out_channels, stride=1):
    # since BN is used, bias is not necessary
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResBlock2D(nn.Module):
    """ 2D residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv_33(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_33(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout2d(p=p)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dp(out)

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
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                            stride=2, padding=0)

    def forward(self, skip, x):
        """ skip is 3D volume and x is 2D slice, central slice of skip is concatenated with x """
        central_inx = skip.size(2) // 2
        skip_slice = skip[:, :, central_inx]

        out = self.transconv(x)
        out = torch.cat([skip_slice, out], 1)

        return out

class ResUNet(nn.Module):
    """ Res UNet class """

    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))

        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        self.dim_reg = ((self.input_size // scale) ** 2) * bottleneck + (self.input_size ** 2) * out_channels
        self.conv_reg = nn.Linear(self.dim_reg, 1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

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
        out = self.conv_n11(out) # fuse several slices in the bottleneck layer
        assert out.size(2) == 1, "there should be only 1 slice in the output of bottleneck layer, but got {}"\
            .format(out.size(2))
        reg1 = out.view(out.size(0), -1)

        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
            else:
                out = self.TransUpBlocks[b_inx](skip, out)

            out = self.BlocksUp[b_inx](out)
            # print(out.size())

        output = self.fl(out)
        reg2 = F.softmax(output, dim=1)
        reg2 = reg2.view(reg2.size(0), -1)  # with size [B, K]
        reg = torch.cat([reg1, reg2], 1)
        assert reg.size(1) == self.dim_reg, "tensor size should be the same as dim_reg"
        # reg = torch.log(torch.exp(self.conv_reg(reg)) + 1)
        reg = torch.squeeze(self.conv_reg(reg), 1)

        return output, reg

def ResUNet28(in_channels, out_channels, n_slices=63, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32, 64, 128, 256, 512], up_blocks = [512, 256, 128, 64, 32], bottleneck = 1024, p=p)

def ResUNet23(in_channels, out_channels, n_slices=31, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32, 64, 128, 256], up_blocks = [256, 128, 64, 32], bottleneck = 512, p=p)

def ResUNet18(in_channels, out_channels, n_slices=15, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32, 64, 128], up_blocks = [128, 64, 32], bottleneck = 256, p=p)

if __name__ == "__main__":
    in_channels = 1
    out_channels = 3
    n_slices = 15
    input_size = 96
    unet = ResUNet18(in_channels, out_channels, n_slices=n_slices, input_size=input_size)
    print(unet)
    x = torch.FloatTensor(6, in_channels, n_slices, input_size, input_size)  # the smallest patch size is 12 * 12
    y = unet(x)