import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import GetTrainingPairs  # Import your data loader
from generator import Generator
from discriminator import Discriminator
from utils.losses import TripletLoss  # Import your loss function
from torch.optim import Adam
from torchvision import transforms
from utils.transforms import create_input_transforms, create_pair_transforms
import os
import matplotlib.pyplot as plt
from PIL import Image

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                      range_vignette=(0.1, 1.5),
                                      std_cap=0.08
                                      )
pair_transforms = create_pair_transforms(flip_prob=0.5)

dataset_path = os.getcwd() + '/../Data/'

# Hyperparameters
batch_size = 16
learning_rate = 0.0002
num_epochs = 100

target_size=(256, 256)

pair_transforms = create_pair_transforms(target_size, flip_prob=0.0)
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                      range_vignette=(0.2, 1.0),
                                      std_cap=0.08
                                      )


# Initialize models and loss function
generator = Generator(input_channels=3, output_channels=3).to(device)
discriminator = Discriminator(input_channels=3).to(device)
triplet_loss_fn = TripletLoss().to(device)

# Initialize optimizers
optimizer_G = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        # Transfer data to device
        real_images = batch['B'].to(device)
        distorted_images = batch['A'].to(device)

        # Train Generator
        optimizer_G.zero_grad()

        # Generate enhanced images
        generated_images = generator(distorted_images)

        # Calculate triplet loss
        triplet_loss = triplet_loss_fn(real_images, generated_images, distorted_images)

        # Backward pass and optimization for generator
        triplet_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Forward pass for real images
        real_validity = discriminator(real_images)

        # Forward pass for generated images
        generated_validity = discriminator(generated_images.detach())

        # Discriminator loss
        d_loss = 0.5 * (torch.mean((real_validity - 1) ** 2) + torch.mean(generated_validity ** 2))

        # Backward pass and optimization for discriminator
        d_loss.backward()
        optimizer_D.step()

        # Print training information
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {triplet_loss.item()}]")

    # Save models or generate sample images at the end of each epoch, if needed
    # ...

# Save final models after training
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
