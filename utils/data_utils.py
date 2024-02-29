import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GetTrainingPairs(Dataset):
    def __init__(self, root, dataset_name, input_transforms_=None, pair_transforms=None):
        self.input_transforms = transforms.Compose(input_transforms_) if input_transforms_ is not None else None
        self.pair_transforms = transforms.Compose(pair_transforms) if pair_transforms is not None else None
        self.filesA, self.filesB = self.get_file_paths(root, dataset_name)
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = transforms.ToTensor()(Image.open(self.filesA[index % self.len]))
        img_B = transforms.ToTensor()(Image.open(self.filesB[index % self.len]))

        if self.pair_transforms:
            img_A = self.pair_transforms(img_A)
            img_B = self.pair_transforms(img_B)

        if self.input_transforms:
            img_A = self.input_transforms(img_A)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        filesA, filesB = [], []

        if dataset_name == 'EUVP':
            sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            for sd in sub_dirs:
                filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
                filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))

        elif dataset_name == 'UFO-120':
            filesA = sorted(glob.glob(os.path.join(root, 'lrd') + "/*.*"))
            filesB = sorted(glob.glob(os.path.join(root, 'hr') + "/*.*"))

        return filesA, filesB

class GetValImage(Dataset):
    def __init__(self, root, dataset_name, transforms_=None, sub_dir='validation'):
        self.transform = transforms.Compose(transforms_)
        self.files = self.get_file_paths(root, dataset_name)
        self.len = len(self.files)

    def __getitem__(self, index):
        img_val = Image.open(self.files[index % self.len])
        img_val = self.transform(img_val)
        return {"val": img_val}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        files = []

        if dataset_name == 'EUVP':
            sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            for sd in sub_dirs:
                files += sorted(glob.glob(os.path.join(root, sd, 'validation') + "/*.*"))

        elif dataset_name == 'UFO-120':
            files = sorted(glob.glob(os.path.join(root, 'lrd') + "/*.*"))

        return files
