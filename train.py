import json
from functools import partial
import sys

# only use gpu 3
import os

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from models.HookedRecursiveCNN import HookedRecursiveCNN
from modules.div_norm_channel import DivisiveNormChannel
from models.adaptation import Adaptation
from temporal_datasets.noisy_dataloader import NoisyTemporalDataset
from modules.divisive_norm import DivisiveNorm
from modules.divisive_norm_group import DivisiveNormGroup
from modules.exponential_decay import ExponentialDecay
from modules.lateral_recurrence import LateralRecurrence
from utils.transforms import RandomRepeatedNoise, MeanFlat
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
                    timestep_transforms = [noise_transformer] * 5 + [MeanFlat()] + [first_in_line_transformer] + 2 * [noise_transformer_test]
                    # Create instances of the Fashion MNIST dataset
                    train_sets.append(NoisyTemporalDataset('train', dataset=dataset,
                                                           transform=transform,
                                                           img_to_timesteps_transforms=timestep_transforms))

                    # for the val set to power evals, we need to explicitly add contrast information to every sample
                    cs = [0.2, 0.4, 0.6, 0.8, 1.0] if contrast == 'random' else [contrast]
                    for c in cs:
                        noise_transformer = RandomRepeatedNoise(contrast=c,
                                                                repeat_noise_at_test=repeat_noise_at_test)
                        noise_transformer_test = partial(noise_transformer, stage='test')
                        first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)
                        timestep_transforms = [noise_transformer] * 5 + [MeanFlat()] + [first_in_line_transformer] + 2 * [noise_transformer_test]
                        test_sets.append(EvalDataWrapper(NoisyTemporalDataset('test', dataset=dataset,
                                                                              transform=transform,
                                                                              img_to_timesteps_transforms=timestep_transforms),
                                                         contrast=c, rep_noise=repeat_noise_at_test)
                                         )
                train_dataset = torch.utils.data.ConcatDataset(train_sets)
                test_dataset = torch.utils.data.ConcatDataset(test_sets)
            else:
                train_sets = []
                test_sets = []
                for contrast in [0.2, 1.0]:
                    noise_transformer = RandomRepeatedNoise(contrast=contrast, repeat_noise_at_test=repeat_noise, noise_component='both')
                    noise_transformer_test = partial(noise_transformer, stage='test')
                    first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)

                    timestep_transforms = [MeanFlat()] + [noise_transformer] * 6 + [MeanFlat()] + [
                        first_in_line_transformer] + [noise_transformer_test]
                    timestep_transforms = [noise_transformer] + [MeanFlat()] + [first_in_line_transformer]
                    # Create instances of the Fashion MNIST dataset
                    train_dataset = NoisyTemporalDataset('train', dataset=dataset, transform=transform,
                                                         img_to_timesteps_transforms=timestep_transforms)
                    train_sets.append(train_dataset)

                for noise in [True, False]:
                    # for the val set to power evals, we need to explicitly add contrast information to every sample
                    cs = [0.2, 0.4, 0.6, 0.8, 1.0]
                    for c in cs:
                        noise_transformer = RandomRepeatedNoise(contrast=c,
                                                                repeat_noise_at_test=noise,
                                                                noise_component='both')
                        noise_transformer_test = partial(noise_transformer, stage='test')
                        first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)
                        timestep_transforms = [noise_transformer] + [MeanFlat()] + [first_in_line_transformer]
                        test_sets.append(EvalDataWrapper(NoisyTemporalDataset('test', dataset=dataset,
                                                                              transform=transform,
                                                                              img_to_timesteps_transforms=timestep_transforms),
                                                         contrast=c, rep_noise=noise))
                test_dataset = torch.utils.data.ConcatDataset(test_sets)
                train_dataset = torch.utils.data.ConcatDataset(train_sets)

            # Create the DataLoaders for the Fashion MNIST dataset
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3)

            # Print the length of the DataLoader
            print(f'Train DataLoader: {len(train_loader)} batches')
            # print(f'Test DataLoader: {len(test_loader)} batches')

            # Visualize the first batch of images
            visualize_first_batch_with_timesteps(train_loader, 8)

            t_steps = next(iter(train_loader))[0].shape[1]

            for num_epoch in config["num_epochs"]:
                hooked_model = HookedRecursiveCNN(t_steps=t_steps, layer_kwargs=layer_kwargs,
                                                  adaptation_module=adaptation_module,
                                                  adaptation_kwargs=adaptation_kwargs)
                l, cache = hooked_model.run_with_cache(next(iter(train_loader))[0])
                model = Adaptation(hooked_model, lr=config["lr"], )

                # load checkpoint
                # checkpoint_path = f"learned_models/pretrained_divnormchannel_DivisiveNormChannel_contrast_0.2_repeat_noise_True_epoch_8.ckpt"
                # checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load checkpoint
                # state_dict = checkpoint['state_dict']  # Extract state dict

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
                contrast = '1.0'
                tb_logger = TensorBoardLogger("lightning_logs",
                                              name=f'{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}',
                                              version=f'generalization_{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}')

                # wandb.init(project='ai-thesis', config=config, entity='ai-thesis', name=f'{config["log_name"]}_{config["adaptation_module"]}_c_{contrast}_rep_{repeat_noise}_ep_{num_epoch}')
                wandb_logger = pl.loggers.WandbLogger(project='ai-thesis', config=config,
                                                      name=f'adapter_contrast_cifar_{config["adaptation_module"]}_c_{contrast}_rep_{repeat_noise}_ep_{num_epoch}_{config["log_name"]}')

                trainer = pl.Trainer(max_epochs=num_epoch, logger=wandb_logger)
                hooked_model.cuda()

                # test_results = trainer.test(model, dataloaders=test_loader)
                wandb_logger.watch(hooked_model, log='all', log_freq=1000)

                trainer.test(model, test_loader)
                trainer.fit(model, train_loader, test_loader, )

                # test
                test_results = trainer.test(model, dataloaders=test_loader)
                print(f'Contrast {contrast}, repeat_noise {repeat_noise}: {test_results[0]["test_acc"]}')
                logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': repeat_noise,
                                    'test_acc': test_results[0]["test_acc"]})
                logger.save()

                trainer.save_checkpoint(
                    f'learned_models/adapter_contrast_cifar_{config["adaptation_module"]}_contrast_{contrast}_repeat_noise_{repeat_noise}_epoch_{num_epoch}.ckpt')
