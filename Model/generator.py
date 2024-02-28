import torch.nn as nn
from utils.modules import ConvolutionalBlock, AttentionBlock, ResidualBlock

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_features=64, num_residual_blocks=8):
        super(Generator, self).__init__()

        # Encoder
        self.enc1 = ConvolutionalBlock(input_channels, num_features, stride=1)
        self.enc2 = ConvolutionalBlock(num_features, num_features * 2)
        self.enc3 = ConvolutionalBlock(num_features * 2, num_features * 4)

        # Attention Module
        self.attention = AttentionBlock(num_features * 4)

        # Decoder with Residual Blocks
        self.dec1 = ResidualBlock(num_features * 4, num_features * 2)
        self.dec2 = ResidualBlock(num_features * 2, num_features)
        self.dec3 = ConvolutionalBlock(num_features, output_channels, activation='tanh', normalization=None)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Attention Module
        attention_output = self.attention(x3)

        # Decoder with Residual Blocks
        x4 = self.dec1(attention_output)
        x5 = self.dec2(x4)
        reconstructed_image = self.dec3(x5)

        return reconstructed_image

