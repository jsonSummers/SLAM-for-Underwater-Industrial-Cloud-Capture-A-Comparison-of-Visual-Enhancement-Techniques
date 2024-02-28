import torch.nn as nn
from utils.modules import ConvolutionalBlock

class Discriminator(nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(Discriminator, self).__init__()

        # Convolutional Blocks with Downsampling
        self.conv1 = ConvolutionalBlock(input_channels, num_features, normalization=None)
        self.conv2 = ConvolutionalBlock(num_features, num_features * 2)
        self.conv3 = ConvolutionalBlock(num_features * 2, num_features * 4)
        self.conv4 = ConvolutionalBlock(num_features * 4, num_features * 8)

        # Output Layer
        self.output_layer = nn.Conv2d(num_features * 8, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        validity_response = self.output_layer(x)
        return validity_response
