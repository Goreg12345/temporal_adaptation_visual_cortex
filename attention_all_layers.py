import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.transforms import Identity
import torchvision.transforms.functional as TF

from modules.lateral_recurrence import LateralRecurrence
from modules.exponential_decay import ExponentialDecay
from modules.divisive_norm import DivisiveNorm
from modules.divisive_norm_group import DivisiveNormGroup
from modules.div_norm_channel import DivisiveNormChannel
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import pytorch_lightning as pl
import json
from models.adaptation import Adaptation
from HookedRecursiveCNN import HookedRecursiveCNN


class TemporalAugmentedDataset(Dataset):
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

    #def augment_image(self, image_tensor):
    #    image_tensor = image_tensor.clamp(0, 1)
    #    image_tensor = transforms.ToPILImage()(image_tensor)
    #    # Define the transformations
    #    transform = transforms.Compose([
    #        transforms.RandomAffine(degrees=(-8, 8), translate=(0, 0.1), scale=(0.9, 1.1), shear=(-5, 5)),
    #        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    #        transforms.ToTensor(),
    #        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)  # Adding random noise
    #    ])
#
    #    # Apply the transformations
    #    augmented_image = transform(image_tensor)
#
    #    # Clamp the values to be between 0 and 1
    #    augmented_image = torch.clamp(augmented_image, 0, 1)
#
    #    return augmented_image

    def augment_image(self, image_tensor):
        # Random affine parameters
        angle = random.uniform(-8, 8)  # degrees
        translate = [random.uniform(0, 3), random.uniform(0, 3)]  # x and y translation
        scale = random.uniform(0.95, 1.05)
        shear = random.uniform(-5, 5)

        # Apply affine transformation
        augmented_image = TF.affine(image_tensor, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)

        # Add random noise
        noise = torch.randn_like(augmented_image) * 0.1
        augmented_image = augmented_image + noise

        # Adjust brightness or contrast
        if random.random() > 0.5:
            augmented_image = TF.adjust_brightness(augmented_image, brightness_factor=random.uniform(0.8, 1.2))
        else:
            augmented_image = TF.adjust_contrast(augmented_image, contrast_factor=random.uniform(0.8, 1.2))

        # Clamp the values to be between 0 and 1
        augmented_image = torch.clamp(augmented_image, 0, 1)

        return augmented_image

    def to_full_img(self, imgs, full_img):
        # imgs = location channel width height
        #full_img = torch.empty((1, 28 * 2, 28 * 2))
        for i in range(4):
            x, y = self.int_to_coordinate(i)
            full_img[..., x:x+28, y:y+28] = self.augment_image(imgs[i] + 0.1)
        full_img = full_img.clip(0, 1)
        return full_img

    def __getitem__(self, index):
        img_timesteps = torch.empty(20, 1, 28 * 2, 28 * 2)
        labels = list()

        # sample 4 ints between 0 and 20
        img_onsets = np.random.randint(0, 20, 3)
        img_onsets = np.append(img_onsets, 0)
        img_locations = np.random.choice(4, 4, replace=False)
        # prev_img = torch.zeros((1, 28 * 2, 28 * 2)) + 0.5
        prev_imgs = torch.zeros((4, 1, 28, 28))
        n_image = 0
        for i, trans_func in enumerate(self.img_to_timesteps_transforms):

            if i in img_onsets:
                # count number of times it's in the list
                count = img_onsets.tolist().count(i)
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

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

class EvalDataWrapper(Dataset):
    """Simple Wrapper that adds contrast and repeated noise information to the dataset to power
    the evaluation metrics"""

    def __init__(self, dataset, contrast, rep_noise):
        self.dataset = dataset
        self.contrast = float(contrast)
        self.rep_noise = bool(rep_noise)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if type(idx) == int:
            idx = torch.tensor([idx])
        contrast = torch.full_like(idx, self.contrast, dtype=torch.float)
        rep_noise = torch.full_like(idx, self.rep_noise, dtype=torch.bool)
        return x, y, contrast, rep_noise

import resource

# Get the current soft limit on file descriptors
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
desired_limit = 20240
# Try to set a new soft limit (this cannot exceed the hard limit)
new_soft_limit = min(hard_limit, desired_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))


if __name__=='__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    eye = Identity()

    timestep_transforms = [eye] * 20
    # Create instances of the Fashion MNIST dataset
    train_dataset = TemporalAugmentedDataset('train', transform=transform,
                                    img_to_timesteps_transforms=timestep_transforms)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=190, worker_init_fn=worker_init_fn,
                              pin_memory=True, pin_memory_device='cuda', prefetch_factor=4)

    from utils.visualization import visualize_first_batch_with_timesteps

    visualize_first_batch_with_timesteps(train_loader, 8)


    test_loader = DataLoader(EvalDataWrapper(train_dataset, contrast=1, rep_noise=False), batch_size=200, shuffle=True,
                             num_workers=190, worker_init_fn=worker_init_fn)

    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset = config["dataset"]
    if config["dataset"] == 'fashion_mnist':
        layer_kwargs = config["layer_kwargs_fmnist"]
    elif config["dataset"] == 'cifar10':
        layer_kwargs = config["layer_kwargs_cifar10"]

    # Define transforms for data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    logger = CSVLogger(config["log_dir"], name=config["log_name"])

    if config["adaptation_module"] == 'LateralRecurrence':
        adaptation_module = LateralRecurrence
        adaptation_kwargs = config["adaptation_kwargs_lateral"]
    elif config["adaptation_module"] == 'ExponentialDecay':
        adaptation_module = ExponentialDecay
        adaptation_kwargs = config["adaptation_kwargs_additive"]
    elif config["adaptation_module"] == 'DivisiveNorm':
        adaptation_module = DivisiveNorm
        adaptation_kwargs = config["adaptation_kwargs_div_norm"]
    elif config["adaptation_module"] == 'DivisiveNormGroup':
        adaptation_module = DivisiveNormGroup
        adaptation_kwargs = config["adaptation_kwargs_div_norm_group"]
    elif config["adaptation_module"] == 'DivisiveNormChannel':
        adaptation_module = DivisiveNormChannel
        adaptation_kwargs = config["adaptation_kwargs_div_norm_channel"]
    else:
        raise ValueError(f'Adaptation module {config["adaptation_module"]} not implemented')

    t_steps = 20

    num_epoch = 50

    layer_kwargs = [{'in_channels': 1, 'out_channels': 32, 'kernel_size': 5},
                    {'in_channels': 32, 'out_channels': 32, 'kernel_size': 5},
                    {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3},
                    {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3},
                    {'in_features': 128, 'out_features': 1024}]
    adaptation_kwargs = [
        {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True, "sigma_init": 0.3,
         "train_sigma": True, 'sqrt': True},
        {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True,
         "sigma_init": 0.3, "train_sigma": True, 'sqrt': True},
        {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True,
         "sigma_init": 0.3, "train_sigma": True, 'sqrt': True},
        {"epsilon": 1e-8, "K_init": 1.0, "train_K": False, "alpha_init": -2000000.0, "train_alpha": False,
         "sigma_init": 1.0, "train_sigma": False},
        {"epsilon": 1e-8, "K_init": 1.0, "train_K": False, "alpha_init": 0.0, "train_alpha": False, "sigma_init": 1.0,
         "train_sigma": False}
    ]

    adaptation_kwargs = [
        {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
        {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
        {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
        {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
        {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False}
      ]

    hooked_model = HookedRecursiveCNN(t_steps=t_steps, layer_kwargs=layer_kwargs,
                                      adaptation_module=adaptation_module,
                                      adaptation_kwargs=adaptation_kwargs, decode_every_timestep=True)
    model = Adaptation(hooked_model, lr=config["lr"], contrast_metrics=False)

    contrast = 'random'
    tb_logger = TensorBoardLogger("lightning_logs",
                                  name=f'augmented_attn_all_layers_{config["adaptation_module"]}_002_sqrt',
                                  version=f'attn_all_layers_{config["adaptation_module"]}_001')

    # wandb.init(project='ai-thesis', config=config, entity='ai-thesis', name=f'{config["log_name"]}_{config["adaptation_module"]}_c_{contrast}_rep_{repeat_noise}_ep_{num_epoch}')
    wandb_logger = pl.loggers.WandbLogger(project='ai-thesis', config=config,
                                          name=f'augmented_attn_all_layers_{config["adaptation_module"]}_c_{contrast}_ep_{num_epoch}_{config["log_name"]}')

    trainer = pl.Trainer(max_epochs=num_epoch, logger=wandb_logger)
    # test_results = trainer.test(model, dataloaders=test_loader)
    wandb_logger.watch(hooked_model, log='all', log_freq=1000)

    trainer.fit(model, train_loader, test_loader)

    # test
    # test_results = trainer.test(model, dataloaders=train_loader)
    # logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': 'n/a',
    #                     'test_acc': test_results[0]["test_acc"]})
    # logger.save()

    trainer.save_checkpoint(
        f'learned_models/augmented_attn_all_layers_{config["adaptation_module"]}_contrast_{contrast}_epoch_{num_epoch}.ckpt')