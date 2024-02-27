import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerBlock, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, dim_feedforward, dropout)

    def forward(self, x):
        return self.transformer(x)


class CustomCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size, stride, padding, transformer_params):
        super(CustomCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            CustomTransformerBlock(**transformer_params),
        )

    def forward(self, x):
        return self.cnn(x)
