import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import GetTrainingPairs
from torchvision import transforms
from utils.transforms import create_input_transforms, create_pair_transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from PIL import Image

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = os.getcwd() + '\\..\\Data\\Paired'
#dataset_path = os.getcwd() + '/../Data/'

target_size=(256, 256)

pair_transforms = create_pair_transforms(target_size, flip_prob=0.0)
input_transforms = create_input_transforms(ratio_min_dist=0.5,
                                      range_vignette=(0.2, 1.0),
                                      std_cap=0.08
                                      )

test_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP', input_transforms_=input_transforms,
                                pair_transforms=pair_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

print("loader created")
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
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original Distorted Image
axes[0].imshow(to_pil_image(distorted_image.cpu()))
axes[0].set_title('Original Distorted Image')
axes[0].axis('off')


# Original Undistorted Image
axes[1].imshow(to_pil_image(undistorted_image.cpu()))
axes[1].set_title('Original Undistorted Image')
axes[1].axis('off')
print("about to show images")
plt.show()