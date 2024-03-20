# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Enhancer, Discriminator, ModelConfig
from utils.losses import adversarial_loss, l1_loss, content_loss, poly_loss
from utils.data_utils import GetTrainingPairs
from utils.transforms import create_pair_transforms, create_input_transforms
import os
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
print(torch.cuda.is_available())
torch.cuda.empty_cache()

#dataset_path = os.getcwd() + '\\..\\Data\\Paired'
dataset_path = os.getcwd() + '/../Data/Paired'
#dataset_path = os.getcwd() + '/../Data'
print("cwd is:" + dataset_path)

# Hyperparameters
target_size = (256, 256)
config = ModelConfig(in_channels=3,
                     out_channels=3,
                     num_filters=32,
                     kernel_size=4,
                     stride=2)
batch_size = 32
learning_rate = 0.0003
num_epochs = 100


# Initialize the models
enhancer = Enhancer(config).to(device)
discriminator = Discriminator(config).to(device)

vgg_weights_path = './vgg19-dcbb9e9d.pth'
vgg_model = models.vgg19(pretrained=False)
vgg_model.load_state_dict(torch.load(vgg_weights_path))
vgg_model.eval().to(device)


# Define optimization criteria and optimizers
criterion_adversarial = nn.BCEWithLogitsLoss()


optimizer_enhancer = optim.Adam(enhancer.parameters(), lr=learning_rate)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)


pair_transforms = create_pair_transforms(target_size, flip_prob=0.5)
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                           range_vignette=(0.1, 1.0),
                                           std_cap=0.05
                                           )

# Initialize data loader
train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP',
                                 input_transforms_=input_transforms, pair_transforms=pair_transforms)

dataset_length = len(train_dataset)
print(f"Number of samples in the dataset: {dataset_length}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("training data loaded")


def inverse_normalize(image):
    # Assuming the original range was [0, 1]
    image = image * 255.0
    # Clip values to ensure they are within valid range [0, 255]
    image = torch.clip(image, 0, 255).cpu()
    # Convert to uint8 if necessary
    #image = image.dtype(torch.uint8)
    return image


checkpoint_frequency = 5

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        input_images, target_images = batch['input'].to(device), batch['target'].to(device)

        # Zero the gradients for both the enhancer and discriminator
        optimizer_enhancer.zero_grad()
        optimizer_discriminator.zero_grad()

        # Enhancer forward pass
        enhanced_images = enhancer(input_images)
        adv_loss = adversarial_loss(discriminator(enhanced_images), True)
        l1_loss_val = l1_loss(enhanced_images, target_images)
        content_loss_val = content_loss(vgg_model, enhanced_images, target_images)
        poly_loss_val = poly_loss(enhanced_images, target_images, 2)

        # Total loss for the enhancer
        enhancer_loss = adv_loss + l1_loss_val + content_loss_val + poly_loss_val

        # Backward pass and optimization for the enhancer
        enhancer_loss.backward()
        optimizer_enhancer.step()

        # Adversarial forward pass
        real_loss = adversarial_loss(discriminator(target_images), True)
        fake_loss = adversarial_loss(discriminator(enhanced_images.detach()), False)
        discriminator_loss = (real_loss + fake_loss) / 2.0

        # Backward pass and optimization for the discriminator
        discriminator_loss.backward()
        optimizer_discriminator.step()

        # Print statistics
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                  f"Generator Loss: {enhancer_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

    # Save enhanced images at the end of each epoch
    with torch.no_grad():
        enhanced_samples = enhancer(input_images)
        side_by_side = torch.cat((input_images.cpu(), enhanced_samples.cpu()), dim=3)
        save_image(side_by_side, f"enhanced_samples_epoch_{epoch}.png", normalize=False)

    if epoch % checkpoint_frequency == 0:
        torch.save(enhancer.state_dict(), f"enhancer_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

# Save the trained models
torch.save(enhancer.state_dict(), "enhancer.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")