import json
from functools import partial
import sys

# only use gpu 3
import os

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from HookedRecursiveCNN import HookedRecursiveCNN
from modules.div_norm_channel import DivisiveNormChannel
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
    if len(sys.argv) > 2:
        config_path = sys.argv[2]
    else:
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

    for contrast in config["contrasts"]:
        for repeat_noise in config["repeat_noises"]:
            if repeat_noise == 'both':
                train_sets = []
                test_sets = []
                for repeat_noise_at_test in [True, False]:
                    noise_transformer = RandomRepeatedNoise(contrast=contrast,
                                                            repeat_noise_at_test=repeat_noise_at_test)
                    noise_transformer_test = partial(noise_transformer, stage='test')
                    first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)

                    timestep_transforms = [MeanFlat()] + [noise_transformer] * 6 + [MeanFlat()] + [
                        first_in_line_transformer] + [noise_transformer_test]
                    timestep_transforms = [noise_transformer] + [MeanFlat()] + [first_in_line_transformer]
                    # Create instances of the Fashion MNIST dataset
                    train_sets.append(NoisyTemporalDataset('train', dataset=dataset, transform=transform,
                                                           img_to_timesteps_transforms=timestep_transforms))
                    test_sets.append(NoisyTemporalDataset('test', dataset=dataset, transform=transform,
                                                          img_to_timesteps_transforms=timestep_transforms))
                train_dataset = torch.utils.data.ConcatDataset(train_sets)
                test_dataset = torch.utils.data.ConcatDataset(test_sets)
            else:
                noise_transformer = RandomRepeatedNoise(contrast=contrast, repeat_noise_at_test=repeat_noise)
                noise_transformer_test = partial(noise_transformer, stage='test')
                first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)

                timestep_transforms = [MeanFlat()] + [noise_transformer] * 6 + [MeanFlat()] + [
                    first_in_line_transformer] + [noise_transformer_test]
                # timestep_transforms = [first_in_line_transformer]
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
            # print(f'Test DataLoader: {len(test_loader)} batches')

            # Visualize the first batch of images
            visualize_first_batch_with_timesteps(train_loader, 8)

            for num_epoch in config["num_epochs"]:
                hooked_model = HookedRecursiveCNN(t_steps=10, layer_kwargs=layer_kwargs,
                                                  adaptation_module=adaptation_module,
                                                  adaptation_kwargs=adaptation_kwargs)
                l, cache = hooked_model.run_with_cache(next(iter(train_loader))[0])
                model = Adaptation(hooked_model, lr=config["lr"], )

                # load checkpoint
                #checkpoint_path = f"learned_models/pretrained_divnormchannel_DivisiveNormChannel_contrast_0.2_repeat_noise_True_epoch_8.ckpt"
                #checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load checkpoint
                #state_dict = checkpoint['state_dict']  # Extract state dict

                # cause pretrained model was trained with scalar adaptation params
                # if adaptation_module == DivisiveNormChannel:
                #     for k, v in state_dict.items():
                #         if 'adapt' in k:
                #             state_dict[k] = torch.ones((adaptation_kwargs[int(k.split('.')[-2])]['n_channels'],), dtype=torch.float32) * v
                # model.load_state_dict(state_dict)

                # only train adaptation layers
                # for param in model.model.conv_layers.parameters():
                #     param.requires_grad = False
                # for param in model.model.fc1.parameters():
                #     param.requires_grad = False
                # for param in model.model.decoder.parameters():
                #     param.requires_grad = False
                # for param in model.model.adapt_layers.parameters():
                #     param.requires_grad = True

                tb_logger = TensorBoardLogger("lightning_logs",
                                           name=f'{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}',
                                           version=f't=3_02_{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}')

                trainer = pl.Trainer(max_epochs=num_epoch, logger=tb_logger)
                hooked_model.cuda()
                # test_results = trainer.test(model, dataloaders=test_loader)

                trainer.fit(model, train_loader, test_loader, )

                # test
                test_results = trainer.test(model, dataloaders=test_loader)
                print(f'Contrast {contrast}, repeat_noise {repeat_noise}: {test_results[0]["test_acc"]}')
                logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': repeat_noise,
                                    'test_acc': test_results[0]["test_acc"]})
                logger.save()
                trainer.save_checkpoint(
                    f'learned_models/t=3_{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}.ckpt')

                print('stop')
