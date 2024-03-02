import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class OneImageTemporalDataset(Dataset):
    def __init__(self, split, dataset='fashion_mnist', transform=None, img_to_timesteps_transforms=None,
                 contrast='random', transforms_fn=None):
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
        self.contrast = contrast
        self.transforms_fn = transforms_fn

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
        img = (img - mean) * contrast  #+ mean
        return img

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img_timesteps = list()
        labels = list()

        prev_img = torch.zeros((1, 28 * 2, 28 * 2)) + 0.5
        for i, trans_func in enumerate(self.img_to_timesteps_transforms):
            if i == 0:
                if self.transform is not None:
                    img = self.transform(img)
                mask = img > 1e-2
                img = trans_func(img, index, target)

                if self.contrast == 'random':
                    rand_contrast = np.random.uniform(0.1, 1)
                else:
                    rand_contrast = self.contrast
                img = self.adjust_contrast(img, rand_contrast)

                new_img = torch.zeros((1, 28 * 2, 28 * 2))
                x, y = 0, 0
                new_img[:, x:x + 28, y:y + 28] = img * mask

                prev_img = prev_img + new_img
                labels.append(
                    target
                )

                img_timesteps.append(prev_img)
            else:
                img_timesteps.append(prev_img)
                labels.append(labels[-1])

                # Stack the augmented images along the timestep dimension
        img_timesteps = torch.stack(img_timesteps, dim=0)
        labels = torch.tensor(labels)
        return img_timesteps, labels

    def __len__(self):
        return len(self.data)


    @staticmethod
    def get_loader(transforms_fn=None, split='train', num_workers=10, batch_size=64):
        import torchvision.transforms as transforms
        from utils.transforms import Identity
        from torch.utils.data import DataLoader
        from utils.visualization import visualize_first_batch_with_timesteps

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        eye = Identity()

        timestep_transforms = [eye] * 20
        one_image_dataset = OneImageTemporalDataset(split, transform=transform,
                                        img_to_timesteps_transforms=timestep_transforms)

        shuffle = True if split=='train' else False
        loader = DataLoader(one_image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        visualize_first_batch_with_timesteps(loader, 8)
        return loader