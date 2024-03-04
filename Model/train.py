import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Generator, Discriminator, ModelConfig
from utils.losses import adversarial_loss, l1_loss, content_loss, triplet_loss  # Adjust the import statements
from utils.data_utils import GetTrainingPairs
from utils.transforms import create_pair_transforms, create_input_transforms
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
print(torch.cuda.is_available())
torch.cuda.empty_cache()

#dataset_path = os.getcwd() + '\\..\\Data\\Paired'
#dataset_path = os.getcwd() + '/../Data/Paired'
dataset_path = os.getcwd() + '/../Data'
print("cwd is:" + dataset_path)

# Hyperparameters
target_size = (256, 256)
config = ModelConfig(in_channels=3, out_channels=3, num_filters=64)
batch_size = 32
learning_rate = 0.001
num_epochs = 64


# Initialize the models
generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)

vgg_weights_path = './vgg19-dcbb9e9d.pth'
vgg_model = models.vgg19(pretrained=False)  # Set pretrained to False because you're providing your own weights
vgg_model.load_state_dict(torch.load(vgg_weights_path))
vgg_model.eval().to(device)


# Define optimization criteria and optimizers
criterion_adversarial = nn.BCEWithLogitsLoss()


optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)


pair_transforms = create_pair_transforms(target_size, flip_prob=0.5)
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                           range_vignette=(0.0, 1.0),
                                           std_cap=0.05
                                           )

# Initialize data loader
train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP',
                                 input_transforms_=input_transforms, pair_transforms=pair_transforms)

dataset_length = len(train_dataset)
print(f"Number of samples in the dataset: {dataset_length}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("training data loaded")

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

        # L1 loss for the generator
        l1_loss_val = l1_loss(generated_images, target_images)

        # Content loss for the generator
        content_loss_val = content_loss(vgg_model, generated_images, target_images)

        # Triplet loss for the encoder-decoder
        triplet_loss_val = triplet_loss(generated_images, target_images, input_images)

        # Total loss for the generator
        generator_loss = adv_loss + l1_loss_val + content_loss_val + triplet_loss_val

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