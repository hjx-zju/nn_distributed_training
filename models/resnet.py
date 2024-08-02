# -*- coding: utf-8 -*-
import math
import torch.nn as nn
from torchsummary import summary
from .evoNorm import EvoNormSample2d as evonorm_s0

__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
def normalization(planes, groups=2, norm_type='evonorm'):
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == 'batchnorm':
        return nn.BatchNorm2d(planes)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(groups, planes)
    elif norm_type == 'evonorm':
        return evonorm_s0(planes)
    else:
        raise NotImplementedError

class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None,groups=2,norm_type='evonorm'):
        super(BasicBlock, self).__init__()
        self.norm_type = norm_type
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = normalization(out_planes,groups,norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = normalization(out_planes,groups,norm_type)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if 'evo' not in self.norm_type:
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if 'evo' not in self.norm_type:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * 4,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif self.dataset == "imagenet":
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(self, block_fn, planes, block_num, stride=1,groups=2,norm_type='evonorm'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                normalization(planes * block_fn.expansion,groups,norm_type),
            )

        layers = []
        layers.append(block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(block_fn(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResNet_cifar(ResNetBase):
    def __init__(self, dataset, resnet_size,groups=2,norm_type='evonorm'):
        super(ResNet_cifar, self).__init__()
        self.dataset = dataset
        self.norm_type = norm_type
        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = normalization(16,groups=groups,norm_type=norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=16, block_num=block_nums,groups=groups,norm_type=norm_type
        )
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=32, block_num=block_nums, stride=2,groups=groups,norm_type=norm_type
        )
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=64, block_num=block_nums, stride=2,groups=groups,norm_type=norm_type
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(
            in_features=64 * block_fn.expansion, out_features=self.num_classes
        )
        self.softmax = nn.LogSoftmax(dim=1)
        # weight initialization based on layer type.
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if 'evo' not in self.norm_type:
            x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x=self.softmax(x)
        return x


def RESNET():
    """Constructs a ResNet-18 model.
    """

 
    model = ResNet_cifar(dataset="cifar10", resnet_size=20,norm_type='evonorm')
   
    return model


if __name__ == '__main__':
    
    # model = CIFAR_NET(3,5,64)
    model=RESNET()
    
    # model=ResNet8()
    model.to("cuda")
    # print summary
    summary(model,(3,32,32),64)