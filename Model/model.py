# model.py

import torch.nn as nn
from utils.modules import ConvBlock, ConvTBlock, ResidualBlock


class ModelConfig:
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - stride) // 2


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.e1 = ConvBlock(config.in_channels, config.num_filters, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)
        self.e2 = ResidualBlock(config.num_filters)
        self.e3 = ConvBlock(config.num_filters, config.num_filters * 2, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)
        self.e4 = ResidualBlock(config.num_filters * 2)
        self.e5 = ConvBlock(config.num_filters * 2, config.num_filters * 4, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)

    def forward(self, x):
        skip1 = self.e1(x)
        skip2 = self.e2(skip1)
        skip3 = self.e3(skip2)
        skip4 = self.e4(skip3)
        x = self.e5(skip4)
        return x, [skip1, skip2, skip3, skip4]


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.d1 = ConvTBlock(config.num_filters * 4, config.num_filters * 2, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)
        self.d2 = ResidualBlock(config.num_filters * 2)
        self.d3 = ConvTBlock(config.num_filters * 2, config.num_filters, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)
        self.d4 = ResidualBlock(config.num_filters)
        self.d5 = ConvTBlock(config.num_filters, config.out_channels, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)

    def forward(self, x, skips):
        x = self.d1(x)
        x = self.d2(x + skips[3])
        x = self.d3(x + skips[2])
        x = self.d4(x + skips[1])
        x = self.d5(x + skips[0])
        return x


class Enhancer(nn.Module):
    def __init__(self, config):
        super(Enhancer, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        encoded, skips = self.encoder(x)
        decoded = self.decoder(encoded, skips)
        return decoded


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            ConvBlock(config.in_channels, 64, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
            ConvBlock(64, 128, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
            ConvBlock(128, 256, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
            ConvBlock(256, 512, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding)
        )

        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        features = self.discriminator(x)
        output = self.final_conv(features)
        return output