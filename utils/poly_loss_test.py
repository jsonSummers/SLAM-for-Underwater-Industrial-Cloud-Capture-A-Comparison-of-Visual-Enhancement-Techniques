import torch
import torch.nn.functional as F
from Model.model import Encoder, ModelConfig  # Import your encoder
from losses import poly_loss, generate_negatives
from transforms import create_poly_loss_transforms, create_pair_transforms, create_input_transforms  # Import your negative transforms
from data_utils import GetTrainingPairs
import os

config = ModelConfig(in_channels=3,
                     out_channels=3,
                     num_filters=32,
                     kernel_size=4,
                     stride=2)

def test_loss():
    dataset_path = os.getcwd() + '/../Data'
    target_size = (256, 256)

    pair_transforms = create_pair_transforms(target_size, flip_prob=0.0)
    input_transforms = create_input_transforms(ratio_min_dist=0.5, range_vignette=(0.1, 1.0), std_cap=0.05)

    # Initialize data loader
    train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP',
                                     input_transforms_=input_transforms, pair_transforms=pair_transforms)

    # Get a pair of images
    target_image = train_dataset.__getitem__(1)['target']
    distorted_image = train_dataset.__getitem__(1)['input']
    target_image = target_image.unsqueeze(0)
    distorted_image = distorted_image.unsqueeze(0)
    print("Target Image Shape:", target_image.shape)
    print("Distorted Image Shape:", distorted_image.shape)

    # Initialize your encoder (replace YourEncoder with your encoder class)
    encoder = Encoder(config)
    ploy_loss_transforms = create_poly_loss_transforms(epoch=1)

    # Set parameters
    num_extreme_negatives = 3
    negative_batch_size = 5
    margin = 1.0

    # Call the loss function
    loss = poly_loss(target_image, distorted_image, encoder, ploy_loss_transforms,
                     num_extreme_negatives, negative_batch_size, margin)

    print("Loss:", loss.item())

if __name__ == "__main__":
    test_loss()



