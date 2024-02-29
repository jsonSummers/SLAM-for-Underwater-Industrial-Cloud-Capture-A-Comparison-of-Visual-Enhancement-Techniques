import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Generator, Discriminator  # Assuming you've saved the models in a file named model.py
from utils.losses import adversarial_loss, triplet_loss
import os
from utils.data_utils import GetTrainingPairs
from utils.transforms import create_pair_transforms, create_input_transforms

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = os.getcwd() + '\\..\\Data\\Paired'
# dataset_path = os.getcwd() + '/../Data/'

target_size = (256, 256)

pair_transforms = create_pair_transforms(target_size, flip_prob=0.5)
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                           range_vignette=(0.2, 1.0),
                                           std_cap=0.05
                                           )

# Hyperparameters
batch_size = 16
learning_rate = 0.0002
num_epochs = 100

# Initialize data loader
train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP',
                                 input_transforms_=input_transforms, pair_transforms=pair_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

