import os
import random
import sys

from temporal_datasets.temporal_augmented_dataset import TemporalAugmentedDataset

if __name__=='__main__':
    device = sys.argv[1]
    adaptation_module = sys.argv[2]
    is_baseline = sys.argv[3]
    is_baseline = True if int(is_baseline) == 1 else False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.transforms import Identity
import torchvision.transforms.functional as TF

from modules.lateral_recurrence import LateralRecurrence
from modules.exponential_decay import ExponentialDecay
from modules.divisive_norm import DivisiveNorm
from modules.divisive_norm_group import DivisiveNormGroup
from modules.div_norm_channel import DivisiveNormChannel
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import json
from models.adaptation import Adaptation
from models.HookedRecursiveCNN import HookedRecursiveCNN


def augment_image(image_tensor):
    # Random affine parameters
    angle = random.uniform(-12, 12)  # degrees
    translate = [random.uniform(0, 4), random.uniform(0, 4)]  # x and y translation
    scale = random.uniform(0.9, 1.1)
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
                              pin_memory=True, pin_memory_device='cuda', prefetch_factor=4, persistent_workers=True)

    test_loader = DataLoader(EvalDataWrapper(train_dataset, contrast=1, rep_noise=False), batch_size=200, shuffle=True,
                             num_workers=190, worker_init_fn=worker_init_fn, pin_memory=True, pin_memory_device='cuda',
                             prefetch_factor=4, persistent_workers=True)

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

    if adaptation_module == 'LateralRecurrence':
        adaptation_module = LateralRecurrence
        adaptation_kwargs = config["adaptation_kwargs_lateral"]
    elif adaptation_module == 'ExponentialDecay':
        adaptation_module = ExponentialDecay
        adaptation_kwargs = config["adaptation_kwargs_additive"]
    elif adaptation_module == 'DivisiveNorm':
        adaptation_module = DivisiveNorm
        adaptation_kwargs = config["adaptation_kwargs_div_norm"]
    elif adaptation_module == 'DivisiveNormGroup':
        adaptation_module = DivisiveNormGroup
        adaptation_kwargs = config["adaptation_kwargs_div_norm_group"]
    elif adaptation_module == 'DivisiveNormChannel':
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
    if (adaptation_module == DivisiveNorm) or (adaptation_module == DivisiveNormChannel):
        adaptation_kwargs = [
            {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True, "sigma_init": 0.3,
             "train_sigma": True, 'sqrt': True},
            {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True,
             "sigma_init": 0.3, "train_sigma": True, 'sqrt': True},
            {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True,
             "sigma_init": 0.3, "train_sigma": True, 'sqrt': True},
            {"epsilon": 1e-8, "K_init": 0.3, "train_K": True, "alpha_init": -2.0, "train_alpha": True,
             "sigma_init": 0.3, "train_sigma": True, 'sqrt': True},
            {"epsilon": 1e-8, "K_init": 1.0, "train_K": False, "alpha_init": 0.0, "train_alpha": False, "sigma_init": 1.0,
             "train_sigma": False}
        ]
    if adaptation_module == DivisiveNormChannel:
        for arg in adaptation_kwargs:
            arg['n_channels'] = 32
    elif adaptation_module == ExponentialDecay:
        if is_baseline:
            adaptation_kwargs = [
                {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False}
            ]
        else:
            adaptation_kwargs = [
                {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
                {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
                {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
                {"alpha_init":  0.5, "train_alpha": True, "beta_init": 1, "train_beta": True},
                {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False}
              ]

    hooked_model = HookedRecursiveCNN(t_steps=t_steps, layer_kwargs=layer_kwargs,
                                      adaptation_module=adaptation_module,
                                      adaptation_kwargs=adaptation_kwargs, decode_every_timestep=True)
    model = Adaptation(hooked_model, lr=config["lr"], contrast_metrics=False)

    contrast = 'random'

    # wandb.init(project='ai-thesis', config=config, entity='ai-thesis', name=f'{config["log_name"]}_{config["adaptation_module"]}_c_{contrast}_rep_{repeat_noise}_ep_{num_epoch}')
    wandb_logger = pl.loggers.WandbLogger(project='ai-thesis', config=config,
                                          name=f'augmented_attn_all_layers_{str(adaptation_module)}_baseline={is_baseline}_c_{contrast}_ep_{num_epoch}_{config["log_name"]}')

    trainer = pl.Trainer(max_epochs=num_epoch, logger=wandb_logger)
    # test_results = trainer.test(model, dataloaders=test_loader)
    wandb_logger.watch(hooked_model, log='all', log_freq=1000)

    trainer.fit(model, train_loader, test_loader)

    # test
    test_results = trainer.test(model, dataloaders=train_loader)
    logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': 'n/a',
                        'test_acc': test_results[0]["test_acc"]})
    logger.save()

    trainer.save_checkpoint(
        f'learned_models/new_augmented_attn_all_layers_{str(adaptation_module)}_baseline={is_baseline}_contrast_{contrast}_epoch_{num_epoch}.ckpt')