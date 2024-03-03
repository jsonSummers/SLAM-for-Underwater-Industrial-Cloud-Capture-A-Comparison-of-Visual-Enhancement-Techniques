import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Generator, Discriminator, ModelConfig  # Assuming you've saved the models in a file named model.py
from utils.losses import adversarial_loss, triplet_loss
import os
from utils.data_utils import GetTrainingPairs
from utils.transforms import create_pair_transforms, create_input_transforms

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
print(torch.cuda.is_available())


dataset_path = os.getcwd() + '\\..\\Data\\Paired'
#dataset_path = os.getcwd() + '/../Data/'

# Hyperparameters
target_size = (256, 256)
config = ModelConfig(in_channels=3, out_channels=3, num_filters=64)
batch_size = 16
learning_rate = 0.0002
num_epochs = 100


# Initialize the models
generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)

# Define optimization criteria and optimizers
criterion_adversarial = nn.BCEWithLogitsLoss()
criterion_triplet = triplet_loss  # You need to define triplet_loss in your losses.py
optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)


pair_transforms = create_pair_transforms(target_size, flip_prob=0.5)
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                           range_vignette=(0.2, 1.0),
                                           std_cap=0.05
                                           )

# Initialize data loader
train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP',
                                 input_transforms_=input_transforms, pair_transforms=pair_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print("training data loaded")
torch.autograd.set_detect_anomaly(True)

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        input_images, target_images = batch['input'].to(device), batch['target'].to(device)

        # Zero the gradients for both the generator and discriminator
        optimizer_generator.zero_grad()
        optimizer_discriminator.zero_grad()

        # Forward pass through the generator
        generated_images = generator(input_images)

        # Adversarial loss for the generator
        adv_loss = adversarial_loss(discriminator(generated_images), True)

        # Triplet loss for the encoder-decoder
        triplet_loss_val = criterion_triplet(generated_images, target_images, input_images)

        # Total loss for the generator
        generator_loss = adv_loss + triplet_loss_val

        # Backward pass and optimization for the generator
        generator_loss.backward()
        optimizer_generator.step()

        # Adversarial loss for the discriminator
        real_loss = adversarial_loss(discriminator(target_images), True)
        fake_loss = adversarial_loss(discriminator(generated_images.detach()), False)
        discriminator_loss = (real_loss + fake_loss) / 2.0

        # Backward pass and optimization for the discriminator
        discriminator_loss.backward()
        optimizer_discriminator.step()

        # Print statistics
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                  f"Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

    # Save generated images at the end of each epoch
    with torch.no_grad():
        generated_samples = generator(input_images)
        save_image(generated_samples, f"generated_samples_epoch_{epoch}.png", normalize=True)

# Save the trained models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")