import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation='leaky_relu', normalization='batch'):
        super(ConvolutionalBlock, self).__init__()
        layers = []

        # Convolutional layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

        # Normalization
        if normalization == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))

        # Activation
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()

        self.query_conv = ConvolutionalBlock(channels, channels // 8, kernel_size=1)
        self.key_conv = ConvolutionalBlock(channels, channels // 8, kernel_size=1)
        self.value_conv = ConvolutionalBlock(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvolutionalBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalization='batch')
        self.conv2 = ConvolutionalBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, normalization='batch')

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
