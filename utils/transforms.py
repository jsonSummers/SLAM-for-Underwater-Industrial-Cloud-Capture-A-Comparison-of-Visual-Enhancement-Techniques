import torch
import torchvision.transforms as transforms
import numpy as np
import random
## https://towardsdatascience.com/image-augmentation-mastering-15-techniques-and-useful-functions-with-python-codes-44c3f8c1ea1f

class AddVignette(object):
    def __init__(self, ratio_min_dist=0.2, range_vignette=(0.2, 0.8), random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, image):
        h, w = image.shape[1:]
        min_dist = np.array([h, w]) / 2 * self.ratio_min_dist * np.random.random()

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[..., None], [1, 1, 3])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        vignette = vignette.transpose((2, 0, 1))  # Transpose to match the shape of X
        image = image * (1 + sign * vignette)

        return image

class GaussianNoise(object):
    def __init__(self, center=0, std_cap=0.1):
        self.center = center
        self.std_cap = std_cap

    def __call__(self, image):
        random_std = np.random.uniform(0, self.std_cap)
        noise = np.random.normal(self.center, random_std, image.shape)
        image = image + noise
        return image

def create_input_transforms(ratio_min_dist=0.2, range_vignette=(0.2, 0.8), std_cap=0.1):
    custom_transforms = [
        transforms.ToTensor(),
        AddVignette(ratio_min_dist, range_vignette),
        GaussianNoise(std_cap),
        #transforms.RandomHorizontalFlip(p=0.5)
    ]
    return transforms.Compose(custom_transforms)

def create_pair_transforms(flip_prob=0.5):
    custom_transforms = [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=flip_prob)
    ]
    return transforms.Compose(custom_transforms)