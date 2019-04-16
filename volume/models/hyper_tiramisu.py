# _*_ coding: utf-8 _*_

""" implemented the Fully Convolution HyperDenseNet for semantic segmentation """

import torch
import torch.nn as nn
from .utils import _initialize_weights

class DenseLayer(nn.Sequential):
    """ Basic dense layer of DenseNet """
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv3d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout3d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, n_streams, in_channels, growth_rate, n_layers, upsample=False):
        """
        :param n_streams: int, number of parallel streams
        :param in_channels: int, number of input channels
        :param growth_rate: int, growth_rate, the same as output channels in each dense layer
        :param n_layers: int, number of layers
        :param upsample: bool, whether to do upsampling or not
        :param dense_hyper: bool, whether use dense hyper skip connection or not
        """

        super().__init__()
        self.upsample = upsample
        self.n_layers = n_layers
        self.n_streams = n_streams

        self.layers = nn.ModuleList([nn.ModuleList([DenseLayer((in_channels + i * growth_rate) * n_streams, growth_rate)
                                     for _ in range(n_streams)]) for i in range(n_layers)])

    def forward(self, x):
        """
        :param x: list of tensors
        :return: list of tensors for each parallel DenseBlock
        """

        # combine list of tensors into a single tensor
        x_com = torch.cat(x, 1)

        # case of up sampling
        if self.upsample:
            new_features = [[None for _ in range(self.n_layers)] for _ in range(self.n_streams)]
            for l_inx in range(self.n_layers):
                outs = [None for _ in range(self.n_streams)]
                for s_inx in range(self.n_streams):
                    outs[s_inx] = self.layers[l_inx][s_inx](x_com)
                    new_features[s_inx][l_inx] = outs[s_inx]

                x_com = torch.cat([x_com, *outs], 1)

            return [torch.cat(new_feature,1) for new_feature in new_features]

        # case of down sampling
        else:
            for l_inx in range(self.n_layers):
                outs = [None for _ in range(self.n_streams)]
                for s_inx in range(self.n_streams):
                    outs[s_inx] = self.layers[l_inx][s_inx](x_com)
                    x[s_inx] = torch.cat([x[s_inx], outs[s_inx]], 1)

                x_com = torch.cat([x_com, *outs], 1)

            return x

class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, theta=1.0):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        out_channels = int(theta * in_channels)
        self.add_module('conv', nn.Conv3d(in_channels, out_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout3d(0.2))
        self.add_module('maxpool', nn.MaxPool3d(2))

    def forward(self, x):
        return super().forward(x)

class TransitionDownBlock(nn.Module):
    def __init__(self, n_streams, in_channels, theta):
        super().__init__()
        self.layers = nn.ModuleList([TransitionDown(in_channels, theta) for _ in range(n_streams)])

    def forward(self, x):
        outs = []
        for stream, layer in zip(x, self.layers):
            outs.append(layer(stream))

        return outs

class TransitionUp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):

        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3), skip.size(4))
        out = torch.cat([out, skip], 1)
        return out

def center_crop(layer, max_depth, max_height, max_width):
    # central crop for 3D tensor
    _, _, d, h, w = layer.size()
    xyz1 = (d - max_depth) // 2
    xyz2 = (h - max_height) // 2
    xyz3 = (w - max_width) // 2
    return layer[:, :, xyz1:(xyz1 + max_depth), xyz2:(xyz2 + max_height), xyz3:(xyz3 + max_width)]

class TransitionUpBlock(nn.Module):
    def __init__(self, n_streams, in_channels):
        super().__init__()
        self.layers = nn.ModuleList([TransitionUp(in_channels) for _ in range(n_streams)])

    def forward(self, xs, skips):
        outs = []
        for x, skip, layer in zip(xs, skips, self.layers):
            outs.append(layer(x, skip))

        return outs

class BottleneckBlock(nn.Sequential):
    def __init__(self, n_streams, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            n_streams, in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


class FirstConvBlock(nn.Module):
    def __init__(self, n_streams, in_channels, out_channels):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=3, stride=1, padding=1, bias=True)
                                       for _ in range(n_streams)])

    def forward(self, x):
        outs = []
        for stream, layer in zip(x, self.layers):
            outs.append(layer(stream))

        return outs

class FinalConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('finalconv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return super().forward(x)

class FCDenseNet(nn.Module):

    def __init__(self, n_streams=2, in_channels=1, down_blocks=(5,5,5),
                 up_blocks=(5,5,5), bottleneck_layers=5, growth_rate=16,
                 out_chans_first_conv=32, n_classes=5, theta=1.0):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.theta = theta

        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##
        self.firstconv = FirstConvBlock(n_streams, in_channels, out_chans_first_conv)
        cur_channels_count = out_chans_first_conv

        ## Down-sampling path ##
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(n_streams, cur_channels_count, growth_rate, down_blocks[i], upsample=False))
            cur_channels_count += (growth_rate*down_blocks[i])

            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDownBlock(n_streams, cur_channels_count, self.theta))
            cur_channels_count = int(self.theta * cur_channels_count)

        ## Bottleneck ##
        self.add_module('bottleneck',BottleneckBlock(n_streams, cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        ## Up-sampling path ##
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)):
            self.transUpBlocks.append(TransitionUpBlock(n_streams, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            if i != len(up_blocks) - 1:
                self.denseBlocksUp.append(DenseBlock(n_streams,
                    cur_channels_count, growth_rate, up_blocks[i],
                        upsample=True))
            else:
                self.denseBlocksUp.append(DenseBlock(n_streams,
                    cur_channels_count, growth_rate, up_blocks[i],
                    upsample=False))
            prev_block_channels = growth_rate*up_blocks[i]

        cur_channels_count += prev_block_channels

        ## final convolution
        self.finalconv = FinalConv(n_streams * cur_channels_count, n_classes)
        _initialize_weights(self)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """ Hyper input as a list of tensors """

        out = self.firstconv(x)
        # print([item.size() for item in out])
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            # print([item.size() for item in out])
        out = self.bottleneck(out)
        # print([item.size() for item in out])

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
            # print([item.size() for item in out])

        out = torch.cat(out, 1)
        out = self.finalconv(out)

        # print(out.size())
        # out = self.softmax(out)
        return out


def FCDenseNet36(in_channel, n_classes, theta, n_streams=2):
    return FCDenseNet(
        n_streams=n_streams, in_channels=in_channel, down_blocks=(4, 4, 4),
        up_blocks=(4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=32, n_classes=n_classes, theta=theta)


def FCDenseNet43(in_channel, n_classes, theta, n_streams=2):
    return FCDenseNet(
        n_streams=n_streams, in_channels=in_channel, down_blocks=(5, 5, 5),
        up_blocks=(5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=32, n_classes=n_classes, theta=theta)


def FCDenseNet52(in_channel, n_classes, theta, n_streams=2):
    return FCDenseNet(
        n_streams=n_streams, in_channels=in_channel, down_blocks=(4, 5, 7),
        up_blocks=(7, 5, 4), bottleneck_layers=12,
        growth_rate=16, out_chans_first_conv=32, n_classes=n_classes, theta=theta)

# networks the original paper provided
def FCDenseNet57(in_channel, n_classes, theta, n_streams=2):
    return FCDenseNet(
        n_streams=n_streams, in_channels=in_channel, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes, theta=theta)


def FCDenseNet67(in_channel, n_classes, theta, n_streams=2):
    return FCDenseNet(
        n_streams=n_streams, in_channels=in_channel, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, theta=theta)


def FCDenseNet103(in_channel, n_classes, theta, n_streams=2):
    return FCDenseNet(
        n_streams=n_streams, in_channels=in_channel, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, theta=theta)

if __name__ == "__main__":
    n_streams = 2
    in_channel = 1
    fcdensenet = FCDenseNet67(n_streams=n_streams, in_channel=in_channel, n_classes=5, theta=1.0)
    x = [torch.FloatTensor(1, 1, 32, 96, 96) for _ in range(n_streams)]
    y = fcdensenet(x)
    # print(y)

