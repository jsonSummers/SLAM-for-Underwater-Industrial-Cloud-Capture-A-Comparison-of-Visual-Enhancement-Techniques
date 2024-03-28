# transforms.py

import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
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
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w), np.linspace(-h / 2, h / 2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)

        # Apply vignette separately to each color channel
        for c in range(image.shape[0]):
            sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
            image[c] = image[c] * (1 + sign * vignette)

        return image


class GaussianNoise(object):
    def __init__(self, center=0, std_cap=0.001):
        self.center = center
        self.std_cap = std_cap

    def __call__(self, image):
        random_std = np.random.uniform(0, self.std_cap)
        noise = np.random.normal(self.center, random_std, image.shape)

        # Clip pixel values to stay within [0, 1]
        image = np.clip(image + noise, 0, 1)

        return image

class ConvertToLAB(object):
    def __call__(self, image):
        # Convert image to LAB color space
        # Assuming image shape: (C, H, W), values in range [0, 1]
        L = 0.3811 * image[0] + 0.5783 * image[1] + 0.0402 * image[2]  # L channel
        M = 0.1967 * image[0] + 0.7244 * image[1] + 0.0782 * image[2]  # M channel
        S = 0.0241 * image[0] + 0.1288 * image[1] + 0.8444 * image[2]  # S channel

        L = (L ** (1 / 2.2)) * 100  # Non-linear gamma correction, then scale to [0, 100]
        M = (M ** (1 / 2.2)) * 100
        S = (S ** (1 / 2.2)) * 100

        # Normalize LAB channels
        L = (L - 50) / 50
        M = (M - 50) / 50
        S = (S - 50) / 50

        lab_image = torch.stack((L, M, S), dim=0)
        return lab_image


def create_input_transforms(ratio_min_dist=0.2, range_vignette=(0.2, 0.8), std_cap=0.1):
    return [
        AddVignette(ratio_min_dist, range_vignette),
        GaussianNoise(std_cap=std_cap),
        ConvertToLAB(),
    ]


def create_pair_transforms(target_size=(256, 256), flip_prob=0.5):
    return [
        # transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=flip_prob)
    ]


def adjust_param(init_val, final_val, epoch, max_epochs):
    return max(init_val - epoch * ((init_val - final_val) / max_epochs), final_val)

def adjust_transforms(epoch, max_epochs=100, constant_after_epoch=None):
    # Define the maximum epochs for adjustment
    max_adjustment_epochs = max_epochs if constant_after_epoch is None else constant_after_epoch

    # Define the parameters for each transformation based on the current epoch
    init_brightness, final_brightness = 0.3, 0.1
    brightness = adjust_param(init_brightness, final_brightness, epoch, max_adjustment_epochs)

    init_contrast, final_contrast = 0.2, 0.1
    contrast = adjust_param(init_contrast, final_contrast, epoch, max_adjustment_epochs)

    init_saturation, final_saturation = 0.2, 0.1
    saturation = adjust_param(init_saturation, final_saturation, epoch, max_adjustment_epochs)

    init_hue, final_hue = 0.35, 0.05
    hue = adjust_param(init_hue, final_hue, epoch, max_adjustment_epochs)

    init_blur_sigma, final_blur_sigma = 2.0, 1.0
    blur_sigma = adjust_param(init_blur_sigma, final_blur_sigma, epoch, max_adjustment_epochs)

    init_rotation_degrees, final_rotation_degrees = 15, 0
    rotation_degrees = adjust_param(init_rotation_degrees, final_rotation_degrees, epoch, max_adjustment_epochs)

    init_affine_degrees, final_affine_degrees = 15, 0
    affine_degrees = adjust_param(init_affine_degrees, final_affine_degrees, epoch, max_adjustment_epochs)

    init_affine_translate, final_affine_translate = 0.1, 0.05
    affine_translate = adjust_param(init_affine_translate, final_affine_translate, epoch, max_adjustment_epochs)

    init_affine_scale, final_affine_scale = 1.1, 1.0
    affine_scale = min(adjust_param(init_affine_scale, final_affine_scale, epoch, max_adjustment_epochs), 1.0)

    init_shear, final_shear = 15, 0
    shear = adjust_param(init_shear, final_shear, epoch, max_adjustment_epochs)

    init_erasing_scale, final_erasing_scale = 0.1, 0.02
    erasing_scale = adjust_param(init_erasing_scale, final_erasing_scale, epoch, max_adjustment_epochs)

    # Construct the transform with adjusted parameters
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, blur_sigma)),
        transforms.RandomRotation(degrees=rotation_degrees),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=affine_degrees, translate=(affine_translate, affine_translate),
                                scale=(affine_scale, 1.0), shear=shear),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.5, scale=(erasing_scale, 0.1), ratio=(0.3, 3.3))
    ])
    return transform

def create_poly_loss_transforms(epoch):
    # Adjust the transforms based on the current epoch
    transforms = adjust_transforms(epoch, max_epochs=100, constant_after_epoch=None)
    return transforms
