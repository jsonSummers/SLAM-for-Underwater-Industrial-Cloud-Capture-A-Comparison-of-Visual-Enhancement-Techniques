import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size, stride, padding):
        super(CustomCNN, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update in_channels for the next layer

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)