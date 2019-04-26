import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleUnit(nn.Module):

    @staticmethod
    def channel_shuffle(x, groups):
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

    @staticmethod
    def _add(residual, x):
        return residual + x

    @staticmethod
    def _concat(residual, x):
        return torch.cat((residual, x), 1)

    def __init__(self, in_channels, out_channels, groups=3, combine='add'):
        super().__init__()
        # figure 2(b) in Paper
        self.groups = groups
        self.combine = combine
        self.combine_func = self._add

        # temp variable
        depthwise_stride = 1
        bottleneck_channels = out_channels // 4

        # figure 2(c) in Paper
        if combine == 'concat':
            self.combine_func = self._concat
            depthwise_stride = 2


        # shuffle unit define
        self.Conv1x1_in = nn.Conv2d(in_channels, bottleneck_channels,
                kernel_size=1, stride=1, groups=groups, bias=False)
        self.Conv1x1_in_bn = nn.BatchNorm2d(bottleneck_channels)
        self.DWConv3x3 = nn.Conv2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=3, stride=depthwise_stride, padding=1,
                groups=bottleneck_channels, bias=False)
        self.DWConv3x3_bn = nn.BatchNorm2d(bottleneck_channels)
        self.Conv1x1_out = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, groups=groups, bias=False)
        self.Conv1x1_out_bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        residual = x
        # figure 2(c) in Paper
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        # compress
        x = self.Conv1x1_in(x)
        x = self.Conv1x1_in_bn(x)
        x = F.relu(x, inplace=True)
        # novel operation of shufflenet
        x = self.channel_shuffle(x, self.groups)
        x = self.DWConv3x3(x)
        x = self.DWConv3x3_bn(x)
        # expand
        x = self.Conv1x1_out(x)
        x = self.Conv1x1_out_bn(x)
        # add residual
        x = self.combine_func(residual, x)
        x = F.relu(x, inplace=True)
        return x


class ShuffleNet(nn.Module):
    """work as a startup"""
    def __init__(self, input_size, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1) # 1/2
        self.su1 = ShuffleUnit(24, 24, groups=3, combine='add')
        self.su2 = ShuffleUnit(24, 48, groups=3, combine='concat') # 1/4
        self.su3 = ShuffleUnit(72, 72, groups=3, combine='add')
        self.su4 = ShuffleUnit(72, 72, groups=3, combine='concat') # 1/8
        self.su5 = ShuffleUnit(144, 144, groups=3, combine='concat') # 1/16
        self.fc = nn.Linear(288 * input_size * input_size // 16 // 16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.su1(x)
        x = self.su2(x)
        x = self.su3(x)
        x = self.su4(x)
        x = self.su5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    inputs = torch.randn(32, 3, 224, 224)
    net = ShuffleNet(224, 3, 196)
    net.train()
    outputs = net(inputs)
    print(net)
