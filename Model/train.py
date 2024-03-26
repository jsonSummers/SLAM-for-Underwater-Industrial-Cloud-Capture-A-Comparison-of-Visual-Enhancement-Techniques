# train.py
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
import numpy as np
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import Enhancer, Discriminator, ModelConfig
from utils.losses import adversarial_loss, l1_loss, content_loss, poly_loss
from utils.data_utils import GetTrainingPairs
from utils.transforms import create_pair_transforms, create_input_transforms, create_poly_loss_transforms

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
#print(torch.cuda.is_available())
torch.cuda.empty_cache()

#dataset_path = os.getcwd() + '\\..\\Data\\Paired'
dataset_path = os.getcwd() + '/../Data/Paired'
#dataset_path = os.getcwd() + '/../Data'
#print("cwd is:" + dataset_path)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
writer = SummaryWriter()

# Define the number of evaluation pairs
num_evaluation_pairs = 5

# Hyperparameters
target_size = (256, 256)
config = ModelConfig(in_channels=3,
                     out_channels=3,
                     num_filters=32,
                     kernel_size=4,
                     stride=2)
batch_size = 48
learning_rate = 0.0003
num_epochs = 100

lambda_adv = 0.5
lambda_l1 = 0.7
lambda_con = 0.3
lambda_poly = 0.5


# Initialize the models
enhancer = Enhancer(config).to(device)
discriminator = Discriminator(config).to(device)


# vgg_weights_path = './vgg19-dcbb9e9d.pth'
# vgg_model = models.vgg19(pretrained=False)
# vgg_model.load_state_dict(torch.load(vgg_weights_path))
# vgg_model.eval().to(device)


# Define optimization criteria and optimizers
criterion_adversarial = nn.BCEWithLogitsLoss()


optimizer_enhancer = optim.Adam(enhancer.parameters(), lr=learning_rate)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)


pair_transforms = create_pair_transforms(target_size, flip_prob=0.0)
input_transforms = create_input_transforms(ratio_min_dist=0.5, range_vignette=(0.1, 1.0), std_cap=0.05)
poly_loss_transforms = create_poly_loss_transforms()

# Initialize data loader
train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP',
                                 input_transforms_=input_transforms, pair_transforms=pair_transforms)

dataset_length = len(train_dataset)
print(f"Number of samples in the dataset: {dataset_length}")

evaluation_indices = np.random.choice(dataset_length, size=num_evaluation_pairs, replace=False)
evaluation_subset = Subset(train_dataset, evaluation_indices)
evaluation_loader = DataLoader(evaluation_subset, batch_size=num_evaluation_pairs, shuffle=False)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("training data loaded")

checkpoint_frequency = 5


def train(num_negatives, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints', 'enhancer'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'final_weights'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            input_images, target_images = batch['input'].to(device), batch['target'].to(device)

            # Zero the gradients
            optimizer_enhancer.zero_grad()
            optimizer_discriminator.zero_grad()

            # Enhancer forward pass
            enhanced_images = enhancer(input_images)
            adv_loss = adversarial_loss(discriminator(enhanced_images), True)
            l1_loss_val = l1_loss(enhanced_images, target_images)
            content_loss_val = content_loss(enhanced_images, target_images)
            poly_loss_val = poly_loss(target_images, enhanced_images, encoder=enhancer.encoder,
                                      negative_transforms=poly_loss_transforms,
                                      num_extreme_negatives=num_negatives, negative_batch_size=(num_negatives + 2))
            enhancer_loss = (lambda_adv * adv_loss) + \
                            (lambda_l1 * l1_loss_val) + \
                            (lambda_con * content_loss_val) + \
                            (lambda_poly * poly_loss_val)

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

            # Logging individual loss values
            writer.add_scalar('Adversarial Loss', adv_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('L1 Loss', l1_loss_val.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Content Loss', content_loss_val.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Poly Loss', poly_loss_val.item(), epoch * len(train_loader) + i)

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                      f"Generator Loss: {enhancer_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

        # Save enhanced images at the end of each epoch
        with torch.no_grad():
            for i, batch in enumerate(evaluation_loader):
                input_images, target_images = batch['input'].to(device), batch['target'].to(device)
                enhanced_samples = enhancer(input_images)
                side_by_side_input = torch.cat((input_images.cpu(), target_images.cpu()), dim=3)
                side_by_side = torch.cat((side_by_side_input, enhanced_samples.cpu()), dim=3)
                save_image(side_by_side, os.path.join(save_path, f"images/evaluation_samples_seed_{seed}_epoch_{epoch}.png"), normalize=False)
                writer.add_images('Original vs. Enhanced', side_by_side, epoch)

            if epoch % checkpoint_frequency == 0:
                torch.save(enhancer.state_dict(), os.path.join(save_path,
                                                               f"checkpoints/enhancer/enhancer_epoch_{epoch}.pth"))
                torch.save(discriminator.state_dict(),
                           os.path.join(save_path, f"checkpoints/discriminator/discriminator_epoch_{epoch}.pth"))

        # Save the trained models
        torch.save(enhancer.state_dict(), os.path.join(save_path, "final_weights/enhancer.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_path, "final_weights/discriminator.pth"))
        writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train models with specified number of negatives.")
    parser.add_argument("num_negatives", type=int, help="Number of negatives.")
    parser.add_argument("save_path", type=str, help="Path to save results.")
    args = parser.parse_args()
    train(args.num_negatives, args.save_path)

if __name__ == "__main__":
    main()