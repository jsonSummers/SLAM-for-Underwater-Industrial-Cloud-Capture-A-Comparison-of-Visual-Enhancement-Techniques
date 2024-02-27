import torch.nn as nn
from utils.cnn_utils import CustomCNN

class TransformerAutoencoder(nn.Module):
    def __init__(self, encoder_cnn_params, encoder_transformer_params, decoder_cnn_params, decoder_transformer_params):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = Encoder(encoder_cnn_params, encoder_transformer_params)
        self.decoder = Decoder(decoder_cnn_params, decoder_transformer_params)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Encoder(nn.Module):
    def __init__(self, cnn_params, transformer_params):
        super(Encoder, self).__init__()
        self.cnn = CustomCNN(**cnn_params, transformer_params=transformer_params)

    def forward(self, x):
        return self.cnn(x)

class Decoder(nn.Module):
    def __init__(self, cnn_params, transformer_params):
        super(Decoder, self).__init__()
        self.cnn = CustomCNN(**cnn_params, transformer_params=transformer_params)

    def forward(self, x):
        return self.cnn(x)