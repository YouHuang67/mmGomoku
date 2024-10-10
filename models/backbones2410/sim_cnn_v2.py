import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


class PreActBlock(nn.Module):

    expansion = 1
    kernel_size = 3
    bottleneck = False

    def __init__(self, in_planes, out_planes, stride=1):
        super(PreActBlock, self).__init__()
        expansion = self.expansion
        kernel_size = self.kernel_size
        if self.bottleneck:
            in_planes = [in_planes, out_planes, out_planes]
            out_planes = [out_planes, out_planes, expansion * out_planes]
            kernel_size = [1, kernel_size, 1]
            stride = [1, stride, 1]
        else:
            in_planes = [in_planes, expansion * out_planes]
            out_planes = [expansion * out_planes, expansion * out_planes]
            kernel_size = [kernel_size, kernel_size]
            stride = [stride, 1]
        self.bn1 = nn.BatchNorm2d(in_planes[0])
        self.conv1 = nn.Conv2d(
            in_planes[0], out_planes[0], kernel_size=kernel_size[0],
            stride=stride[0], padding=(kernel_size[0] - 1) // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes[1])
        self.conv2 = nn.Conv2d(
            in_planes[1], out_planes[1], kernel_size=kernel_size[1],
            stride=stride[1], padding=(kernel_size[1] - 1) // 2, bias=False
        )
        if self.bottleneck:
            self.bn3 = nn.BatchNorm2d(in_planes[2])
            self.conv3 = nn.Conv2d(
                in_planes[2], out_planes[2], kernel_size=kernel_size[2],
                stride=stride[2], padding=(kernel_size[2] - 1) // 2, bias=False
            )

        if stride[-2] != 1 or in_planes[0] != out_planes[-1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes[0], out_planes[-1], kernel_size=1,
                          stride=stride[-2], bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.bottleneck:
            out = self.conv3(F.relu(self.bn3(out)))
        return out + shortcut


class PreBottleneckBlock(PreActBlock):

    expansion = 4
    kernel_size = 3
    bottleneck = True


class KernelOneBlock(PreActBlock):

    expansion = 1
    kernel_size = 1
    bottleneck = False


class KernelOneBottleneckBlock(PreBottleneckBlock):

    expansion = 4
    kernel_size = 1
    bottleneck = True


class PreActResNet(nn.Module):

    def __init__(self, num_blocks, bottleneck, kernel_one_level):
        super(PreActResNet, self).__init__()
        if bottleneck:
            blocks = [KernelOneBottleneckBlock for _ in range(kernel_one_level)]
            blocks += [PreBottleneckBlock for _ in range(4 - len(blocks))]
        else:
            blocks = [KernelOneBlock for _ in range(kernel_one_level)]
            blocks += [PreActBlock for _ in range(4 - len(blocks))]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(blocks[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(blocks[2], 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(blocks[3], 512, num_blocks[3], stride=1)
        self.bn = nn.BatchNorm2d(512 * blocks[-1].expansion)
        self.linear = nn.Conv2d(
            512 * blocks[-1].expansion, 1, kernel_size=1, stride=1, bias=True
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = self.linear(out)
        return out.view(x.size(0), -1)


@MODELS.register_module()
class PreActResNet18(PreActResNet):

    def __init__(self, kernel_one_level=0):
        super(PreActResNet18, self).__init__(
            [2, 2, 2, 2], False, kernel_one_level=kernel_one_level)


@MODELS.register_module()
class PreActResNet34(PreActResNet):

    def __init__(self, kernel_one_level=0):
        super(PreActResNet34, self).__init__(
            [3, 4, 6, 3], False, kernel_one_level=kernel_one_level)


@MODELS.register_module()
class PreActResNet50(PreActResNet):

    def __init__(self, kernel_one_level=0):
        super(PreActResNet50, self).__init__(
            [3, 4, 6, 3], True, kernel_one_level=kernel_one_level)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride,
                 drop_rate=0.0, kernel_size=3,
                 se_layer=False, se_reduction=16):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=(kernel_size - 1) // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False
        )
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
        ) or None
        self.se_layer = se_layer and SELayer(out_planes, se_reduction)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 drop_rate=0.0, kernel_size=3, se_layer=False, se_reduction=16):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, drop_rate, kernel_size,
                                      se_layer, se_reduction)

    def _make_layer(self, block, in_planes, out_planes,
                    nb_layers, stride, drop_rate, kernel_size,
                    se_layer, se_reduction):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, kernel_size,
                                se_layer=se_layer, se_reduction=se_reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor=1, drop_rate=0.0, kernel_one_level=0,
                 se_layer=False, se_reduction=16):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate,
                                   (1 if kernel_one_level > 0 else 3),
                                   se_layer=se_layer, se_reduction=se_reduction)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 1, drop_rate,
                                   (1 if kernel_one_level > 1 else 3),
                                   se_layer=se_layer, se_reduction=se_reduction)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, drop_rate,
                                   (1 if kernel_one_level > 2 else 3),
                                   se_layer=se_layer, se_reduction=se_reduction)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(nChannels[3], 1, kernel_size=1, bias=True)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return self.fc(out).view(x.size(0), -1)


@MODELS.register_module()
class WideResNet16_1(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0):
        super(WideResNet16_1, self).__init__(16, 1, drop_rate, kernel_one_level)


@MODELS.register_module()
class WideResnet16_2(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0):
        super(WideResnet16_2, self).__init__(16, 2, drop_rate, kernel_one_level)


@MODELS.register_module()
class WideResnet16_4(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0):
        super(WideResnet16_4, self).__init__(16, 4, drop_rate, kernel_one_level)


@MODELS.register_module()
class WideResnet40_1(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0):
        super(WideResnet40_1, self).__init__(40, 1, drop_rate, kernel_one_level)


@MODELS.register_module()
class WideResnet40_2(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0):
        super(WideResnet40_2, self).__init__(40, 2, drop_rate, kernel_one_level)


@MODELS.register_module()
class SEWideResnet16_1(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0, se_reduction=1):
        super(SEWideResnet16_1, self).__init__(
            16, 1, drop_rate, kernel_one_level, se_layer=True,
           se_reduction=se_reduction)


@MODELS.register_module()
class SEWideResnet16_2(WideResNet):

    def __init__(self, drop_rate=0.0, kernel_one_level=0, se_reduction=1):
        super(SEWideResnet16_2, self).__init__(
            16, 2, drop_rate, kernel_one_level, se_layer=True,
            se_reduction=se_reduction)
