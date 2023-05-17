import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

class NoisyTemporalDataset(Dataset):
    def __init__(self, split, dataset='fashion_mnist', transform=None, img_to_timesteps_transforms=None):
        """
        Initializes the FashionMNISTNoisyDataset
        :param split: 'train' or 'test'
        :param dataset: name of the dataset
            must be one of the datasets in the huggingface datasets package
        :param transform: transforms to apply to the images
        :param img_to_timesteps_transforms: list of functions
            for every desired timestep, there should be a function in the list that converts
            the image to the desired format
        """
        self.split = split
        self.transform = transform
        self.dataset = load_dataset(dataset, split=split)
        input_col_name = 'img' if 'img' in self.dataset.column_names else 'image'  # because different datasets have different names
        self.data, self.targets = self.dataset[input_col_name], self.dataset['label']
        self.img_to_timesteps_transforms = img_to_timesteps_transforms

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img_timesteps = list()

        for trans_func in self.img_to_timesteps_transforms:
            if self.transform is not None:
                img_transformed = self.transform(img)

            img_timesteps.append(trans_func(img_transformed, index, target))

        # Stack the augmented images along the timestep dimension
        img_timesteps = torch.stack(img_timesteps, dim=0)

        return img_timesteps, target

    def __len__(self):
        return len(self.data)
