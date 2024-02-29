# modules.py

import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Convert input tensor to the same data type as the bias tensor
        x = x.to(self.batchnorm.weight.dtype)

        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvBlock(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(residual)
        x = self.conv2(x + residual)
        return x
