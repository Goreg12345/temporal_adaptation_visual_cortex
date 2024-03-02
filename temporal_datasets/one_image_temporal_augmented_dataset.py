from torch.utils.data import Dataset

import numpy as np
from attention_all_layers import augment_image
import torch
from datasets import load_dataset


class OneImageTemporalAugmentedDataset(Dataset):
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

    def int_to_coordinate(self, index):
        if index == 0:
            return 0, 0
        elif index == 1:
            return 0, 28
        elif index == 2:
            return 28, 0
        elif index == 3:
            return 28, 28
        else:
            raise ValueError('index must be between 0 and 3')

    def adjust_contrast(self, img, contrast):
        mean = img.mean()
        img = (img - mean) * contrast  # + mean
        return img

    def to_full_img(self, imgs, full_img):
        # imgs = location channel width height
        #full_img = torch.empty((1, 28 * 2, 28 * 2))
        for i in range(4):
            x, y = self.int_to_coordinate(i)
            full_img[..., x:x+28, y:y+28] = augment_image(imgs[i] + 0.1)
        full_img = full_img.clip(0, 1)
        return full_img

    def __getitem__(self, index):
        img_timesteps = torch.empty(20, 1, 28 * 2, 28 * 2)
        labels = list()

        # sample 4 ints between 0 and 20
        img_onsets = [0]
        img_locations = [0]
        # prev_img = torch.zeros((1, 28 * 2, 28 * 2)) + 0.5
        prev_imgs = torch.zeros((4, 1, 28, 28))
        n_image = 0
        for i, trans_func in enumerate(self.img_to_timesteps_transforms):

            if i in img_onsets:
                # count number of times it's in the list
                count = img_onsets.count(i)
                cur_labels = list()
                cur_contrasts = list()
                for j in range(count):
                    # sample random image
                    idx = np.random.randint(0, len(self.data))
                    img, target = self.data[idx], int(self.targets[idx])

                    if self.transform is not None:
                        img = self.transform(img)

                    img = trans_func(img, index, target)

                    rand_contrast = np.random.uniform(0.1, 1)
                    img = self.adjust_contrast(img, rand_contrast)

                    prev_imgs[img_locations[n_image]] += img

                    # new_img = torch.zeros((1, 28 * 2, 28 * 2))
                    # x, y = self.int_to_coordinate(img_locations[n_image])
                    # new_img[:, x:x + 28, y:y + 28] = img * mask
                    n_image += 1

                    cur_labels.append(target)
                    cur_contrasts.append(rand_contrast)
                    # prev_img = prev_img + new_img
                labels.append(
                    cur_labels[np.array(cur_contrasts).argmax()]
                )

                self.to_full_img(prev_imgs, img_timesteps[i])
            else:
                self.to_full_img(prev_imgs, img_timesteps[i])
                labels.append(labels[-1])

        labels = torch.tensor(labels)
        return img_timesteps, labels

    def __len__(self):
        return len(self.data)