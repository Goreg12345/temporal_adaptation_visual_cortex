import json
from functools import partial

# only use gpu 3
import os

from HookedRecursiveCNN import HookedRecursiveCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from models.adaptation import Adaptation
from models.additive_adaptation import AdditiveAdaptation
from models.noisy_dataloader import NoisyTemporalDataset
from modules.divisive_norm import DivisiveNorm
from modules.divisive_norm_group import DivisiveNormGroup
from modules.exponential_decay import ExponentialDecay
from modules.lateral_recurrence import LateralRecurrence
from utils.transforms import Identity, RandomRepeatedNoise, Grey, MeanFlat
from utils.visualization import visualize_first_batch_with_timesteps


if __name__ == '__main__':

    with open('config.json', 'r') as f:
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
    else:
        raise ValueError(f'Adaptation module {config["adaptation_module"]} not implemented')

    for contrast in config["contrasts"]:
        for repeat_noise in config["repeat_noises"]:
            noise_transformer = RandomRepeatedNoise(contrast=contrast, repeat_noise_at_test=repeat_noise)
            noise_transformer_test = partial(noise_transformer, stage='test')
            first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)

            timestep_transforms = [MeanFlat()] + [noise_transformer] * 5 + [MeanFlat()] + [first_in_line_transformer] + [noise_transformer_test] * 2
            # Create instances of the Fashion MNIST dataset
            train_dataset = NoisyTemporalDataset('train', dataset=dataset, transform=transform,
                                                 img_to_timesteps_transforms=timestep_transforms)
            test_dataset = NoisyTemporalDataset('test', dataset=dataset, transform=transform,
                                                 img_to_timesteps_transforms=timestep_transforms)

            # Create the DataLoaders for the Fashion MNIST dataset
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=3)

            # Print the length of the DataLoader
            print(f'Train DataLoader: {len(train_loader)} batches')
            #print(f'Test DataLoader: {len(test_loader)} batches')

            # Visualize the first batch of images
            visualize_first_batch_with_timesteps(train_loader, 8)

            for num_epoch in config["num_epochs"]:
                hooked_model = HookedRecursiveCNN(t_steps=10, layer_kwargs=layer_kwargs,
                                                  adaptation_module=adaptation_module,
                                                  adaptation_kwargs=adaptation_kwargs)
                l, cache = hooked_model.run_with_cache(next(iter(train_loader))[0])
                model = Adaptation(hooked_model, lr=config["lr"],)
                trainer = pl.Trainer(max_epochs=num_epoch)
                test_results = trainer.test(model, dataloaders=test_loader)

                trainer.fit(model, train_loader, test_loader, )

                # test
                test_results = trainer.test(model, dataloaders=test_loader)
                print(f'Contrast {contrast}, repeat_noise {repeat_noise}: {test_results[0]["test_acc"]}')
                logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': repeat_noise, 'test_acc': test_results[0]["test_acc"]})
                logger.save()
                trainer.save_checkpoint(f'learned_models/{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}.ckpt')

    print('stop')
