import torch
import torch.optim as optim
from model import TransformerAutoencoder
from utils.data_utils import GetTrainingPairs
from loss import TripletLoss, QuadrupletLoss, Loss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.data_utils import GetTrainingPairs
import torchvision.transforms as transforms
import os

# Set your data root and dataset name
cwd = os.getcwd()
data_root = cwd + '\\Data\\Paired\\'
dataset_name = 'EUVP'

print(data_root)

# Create an instance of the GetTrainingPairs dataset
train_dataset = GetTrainingPairs(root=data_root, dataset_name=dataset_name, transforms_=transforms.ToTensor())

# Create a DataLoader for the training dataset
batch_size = 64  # You can adjust this based on your preferences
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

learning_rate = 0.001
num_epochs = 1

# Set your parameters
encoder_cnn_params = {'in_channels': 3,
                      'out_channels': 64,
                      'num_layers': 3,
                      'kernel_size': 3,
                      'stride': 1,
                      'padding': 1}

encoder_transformer_params = {'d_model': 64,
                              'nhead': 4,
                              'dim_feedforward': 2048,
                              'dropout': 0.1}


decoder_cnn_params = {'in_channels': 64,
                      'out_channels': 3,
                      'num_layers': 3,
                      'kernel_size': 3,
                      'stride': 1,
                      'padding': 1}

decoder_transformer_params = {'d_model': 64,
                              'nhead': 4,
                              'dim_feedforward': 2048,
                              'dropout': 0.1}


model = TransformerAutoencoder(
    encoder_cnn_params, encoder_transformer_params,
    decoder_cnn_params, decoder_transformer_params
)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Instantiate the loss functions
triplet_loss_fn = TripletLoss(margin=1.0)
quadruplet_loss_fn = QuadrupletLoss(margin=1.0)
regular_loss_fn = Loss()

# Instantiate the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch['A'].to(device), batch['B'].to(device)  # Move inputs to GPU
        optimizer.zero_grad()
        outputs = model(inputs)

        # Example: using triplet loss, you can switch to other losses as needed
        loss = triplet_loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()