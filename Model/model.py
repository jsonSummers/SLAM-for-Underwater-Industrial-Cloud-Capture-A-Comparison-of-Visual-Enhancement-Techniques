import torch
import torch.nn as nn
from utils.cnn_utils import CustomCNN

class EncoderModel(nn.Module):
    def __init__(self, cnn_params, embedding_size):
        super(EncoderModel, self).__init__()
        self.cnn = CustomCNN(**cnn_params)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(64, embedding_size)  # Assuming the output channels of CNN are 64

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x