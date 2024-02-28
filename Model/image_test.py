import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import GetTrainingPairs  # Import your data loader
from generator import Generator
from discriminator import Discriminator
from utils.losses import TripletLoss  # Import your loss function
from torch.optim import Adam
from torchvision import transforms
from utils.transforms import create_input_transforms
import os
import matplotlib.pyplot as plt
from PIL import Image

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = os.getcwd() + '/../Data/'

custom_transforms = create_input_transforms(ratio_min_dist=0.5,
                                      range_vignette=(0.1, 1.5),
                                      std_cap=0.08
                                      )

test_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP', transforms_=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

print("loader created")
# Fetch a batch from the data loader
batch = next(iter(test_loader))
print("batch created")

# Extract images from the batch
distorted_image = batch['A'][0]  # Assuming 'A' represents distorted images
undistorted_image = batch['B'][0]  # Assuming 'B' represents undistorted images

print(distorted_image.type())

def to_pil_image(tensor):
    """Convert a tensor to a PIL Image."""
    img = tensor.clone().detach()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    return img

# Display the original and transformed images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original Distorted Image
axes[0].imshow(to_pil_image(distorted_image.cpu()))
axes[0].set_title('Original Distorted Image')
axes[0].axis('off')

# Convert distorted_image to a PIL Image before applying custom_transforms
distorted_image = to_pil_image(distorted_image.cpu())

# Apply custom_transforms on the PIL Image
transformed_image = custom_transforms(distorted_image)
axes[1].imshow(transformed_image.permute(1, 2, 0))
axes[1].set_title('Transformed Image')
axes[1].axis('off')

# Original Undistorted Image
axes[2].imshow(to_pil_image(undistorted_image.cpu()))
axes[2].set_title('Original Undistorted Image')
axes[2].axis('off')

plt.show()