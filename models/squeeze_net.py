# import torch
# import torch.nn as nn
# from torch.nn import functional as F
from torchsummary import summary

# # 定义 Fire 模块 (Squeeze + Expand)
# class Fire(nn.Module):
#     def __init__(self, in_ch, squeeze_ch, e1_ch, e3_ch):  # 声明 Fire 模块的超参数
#         super(Fire, self).__init__()
#         # Squeeze, 1x1 卷积
#         self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1)
#         # # Expand, 1x1 卷积
#         self.expand1 = nn.Conv2d(squeeze_ch, e1_ch, kernel_size=1)
#         # Expand, 3x3 卷积
#         self.expand3 = nn.Conv2d(squeeze_ch, e3_ch, kernel_size=3, padding=1)
#         self.activation = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.activation(self.squeeze(x))
#         x = torch.cat([self.activation(self.expand1(x)),
#                        self.activation(self.expand3(x))], dim=1)
#         return x    
# # 定义简化的 SqueezeNet 模型类 1
# class SqueezeNet1(nn.Module):
#     def __init__(self, num_classes=100):
#         super(SqueezeNet1, self).__init__()
#         self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)  # 3x32x32 -> 96x32x32
#         self.relu = nn.ReLU(inplace=True)
#         self.fire2 = Fire(96, 48, 32, 32)  # 96x32x32 -> 64x32x32
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x32x32 -> 64x16x16
#         self.fire3 = Fire(64, 32, 64, 64)  # 64x16x16 -> 128x16x16
#         self.fire4 = Fire(128, 64, 128, 128)  # 128x16x16 -> 256x16x16
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x16x16 -> 256x8x8
#         self.fire5 = Fire(256, 64, 192, 192)  # 256x8x8 -> 384x8x8
#         self.fire6 = Fire(384, 128, 256, 256)  # 384x8x8 -> 512x8x8
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512x8x8 -> 512x4x4
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))  # 512x4x4 -> 512x1x1
#         self.linear = nn.Linear(512, num_classes)  # 512 -> num_classes
#         self.softmax = nn.LogSoftmax(dim=1)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.fire2(x)
#         x = self.maxpool1(x)  # torch.Size([1, 64, 16, 16])
#         x = self.fire3(x)
#         x = self.fire4(x)
#         x = self.maxpool2(x)  # torch.Size([1, 256, 8, 8])
#         x = self.fire5(x)
#         x = self.fire6(x)
#         x = self.maxpool3(x)    # torch.Size([1, 512, 4, 4])
#         x = self.avg_pool(x)  # torch.Size([1, 512, 1, 1])
#         x = x.view(x.size(0), -1)  # torch.Size([1, 512])
#         x = self.linear(x)  # torch.Size([1, 10])
#         x=self.softmax(x)
#         return x
# # 定义简化的 SqueezeNet 模型类 2
# class SqueezeNet2(nn.Module):
#     def __init__(self, num_classes=100):
#         super(SqueezeNet2, self).__init__()
#         self.num_classes = num_classes
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 3x32x32 -> 64x32x32
#             nn.ReLU(inplace=True),
#             Fire(64, 16, 64, 64),  # 64x32x32 -> 128x32x32
#             nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x32x32 -> 128x16x16
#             Fire(128, 32, 64, 64),  # 128x16x16 -> 128x16x16
#             nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x16x16 -> 128x8x8
#             Fire(128, 64, 128, 128),  # 128x8x8 -> 256x8x8
#             nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 256x8x8 -> 256x4x4
#             Fire(256, 64, 256, 256)  # 256x4x4 -> 512x4x4
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Conv2d(512, self.num_classes, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1)),  # 512x4x4 -> 10x1x1
#         )
#         self.softmax = nn.LogSoftmax(dim=1)
        

#     def forward(self, x):
#         x = self.features(x)  # torch.Size([1, 512, 4, 4])
#         x = self.classifier(x)  # torch.Size([1, 10, 1, 1])
#         x = x.view(x.size(0), -1)  # torch.Size([1, 10])
#         x=self.softmax(x)
        
#         return x

# if __name__ == "__main__":
#     # Test ResNet18
#     model = SqueezeNet1(10)
#     model.to("cuda")
#     # print summary
#     summary(model,(3,32,32),64)


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 19:33:53 2018

@author: akash
"""

# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
import numpy as np
import math


class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f
class CIFAR_NET(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the CIFAR
    classification problem.
    """

    def __init__(self, num_filters, kernel_size, linear_width):
        super().__init__()
        conv_out_width = 32 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)

        self.seq = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc1_indim, linear_width),
            nn.ReLU(inplace=True),
            nn.Linear(linear_width, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.seq(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


if __name__ == '__main__':
    
    model = CIFAR_NET(3,5,64)
    # model=SqueezeNet()
    
    # model=ResNet8()
    model.to("cuda")
    # print summary
    summary(model,(3,32,32),64)