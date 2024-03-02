# only use gpu 3
import os
import sys
from functools import partial

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from models.HookedRecursiveCNN import HookedRecursiveCNN
from models.adaptation import Adaptation
from temporal_datasets.noisy_dataloader import NoisyTemporalDataset
from modules.divisive_norm import DivisiveNorm
from modules.exponential_decay import ExponentialDecay
from utils.transforms import RandomRepeatedNoise, MeanFlat
from utils.visualization import visualize_first_batch_with_timesteps


if __name__ == '__main__':
    # Define transforms for data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    logger = CSVLogger('../experimental_data', name='num_maps_cifar10_30ep')

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


    for adaptation_module, no_time in [(ExponentialDecay, True), (ExponentialDecay, False), (DivisiveNorm, False)]:
        repeat_noise = True
        contrast = 'random'
        if adaptation_module == ExponentialDecay:
            if no_time:
                adaptation_kwargs = [
                    {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                    {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                    {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                    {"alpha_init":  1.0, "train_alpha": False, "beta_init": 1, "train_beta": False}
                ]
            else:
                adaptation_kwargs = [
                    {"alpha_init": 0.5, "train_alpha": True, "beta_init": 2.7, "train_beta": True},
                    {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                    {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False},
                    {"alpha_init": 1.0, "train_alpha": False, "beta_init": 1, "train_beta": False}
                ]
        elif adaptation_module == DivisiveNorm:
            adaptation_kwargs = [
                {"epsilon":  1e-8, "K_init":  0.2, "train_K":  True, "alpha_init":  -2.0, "train_alpha": True, "sigma_init": 0.1, "train_sigma": True},
                {"epsilon":  1e-8, "K_init":  1.0, "train_K":  False, "alpha_init":  -2000000.0, "train_alpha": False, "sigma_init": 1.0, "train_sigma": False},
                {"epsilon":  1e-8, "K_init":  1.0, "train_K":  False, "alpha_init":  -2000000.0, "train_alpha": False, "sigma_init": 1.0, "train_sigma": False},
                {"epsilon":  1e-8, "K_init":  1.0, "train_K":  False, "alpha_init":  0.0, "train_alpha": False, "sigma_init": 1.0, "train_sigma": False}
              ]

        for n_featuremap in [2, 4, 8, 16, 32]:
            # layer_kwargs = [
            #     {"in_channels": 1, "out_channels": n_featuremap, "kernel_size": 5},
            #     {"in_channels": n_featuremap, "out_channels": n_featuremap, "kernel_size": 5},
            #     {"in_channels": n_featuremap, "out_channels": n_featuremap, "kernel_size": 3},
            #     {"in_features": n_featuremap * 4, "out_features": 100}
            # ]
            layer_kwargs = [
                {"in_channels":  3, "out_channels": n_featuremap, "kernel_size": 5},
                {"in_channels":  n_featuremap, "out_channels": n_featuremap, "kernel_size": 5},
                {"in_channels":  n_featuremap, "out_channels": n_featuremap, "kernel_size": 3},
                {"in_features":  n_featuremap * 9, "out_features": 500}
              ]


            test_sets = []
            noise_transformer = RandomRepeatedNoise(contrast=contrast, repeat_noise_at_test=repeat_noise)
            noise_transformer_test = partial(noise_transformer, stage='test')
            first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)

            timestep_transforms = [noise_transformer] + [MeanFlat()] + [first_in_line_transformer]
            # Create instances of the Fashion MNIST dataset
            train_dataset = NoisyTemporalDataset('train', dataset='cifar10', transform=transform,
                                                 img_to_timesteps_transforms=timestep_transforms)

            # for the val set to power evals, we need to explicitly add contrast information to every sample
            cs = [0.2, 0.4, 0.6, 0.8, 1.0] if contrast == 'random' else [contrast]
            for noise in [True, False]:
                for c in cs:
                    noise_transformer = RandomRepeatedNoise(contrast=c,
                                                            repeat_noise_at_test=noise)
                    noise_transformer_test = partial(noise_transformer, stage='test')
                    first_in_line_transformer = partial(noise_transformer, stage='test', first_in_line=True)
                    timestep_transforms = [noise_transformer] + [MeanFlat()] + [first_in_line_transformer]
                    test_sets.append(EvalDataWrapper(NoisyTemporalDataset('test', dataset='cifar10',
                                                                          transform=transform,
                                                                          img_to_timesteps_transforms=timestep_transforms),
                                                     contrast=c, rep_noise=noise))
            test_dataset = torch.utils.data.ConcatDataset(test_sets)

            # Create the DataLoaders for the Fashion MNIST dataset
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=3)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=3)

            # Print the length of the DataLoader
            print(f'Train DataLoader: {len(train_loader)} batches')
            # print(f'Test DataLoader: {len(test_loader)} batches')

            # Visualize the first batch of images
            visualize_first_batch_with_timesteps(train_loader, 8)

            t_steps = next(iter(train_loader))[0].shape[1]

            num_epoch = 30
            hooked_model = HookedRecursiveCNN(t_steps=t_steps, layer_kwargs=layer_kwargs,
                                              adaptation_module=adaptation_module,
                                              adaptation_kwargs=adaptation_kwargs, d_fc=500)
            l, cache = hooked_model.run_with_cache(next(iter(train_loader))[0])
            model = Adaptation(hooked_model, lr=1e-3, )

            wandb_logger = pl.loggers.WandbLogger(project='ai-thesis', config={
                'num_maps': n_featuremap,
                'adaptation_type': f'{str(adaptation_module)}_is_baseline={no_time}',
                'contrast': contrast,
                'epoch': num_epoch,
                'repeat_noise': repeat_noise,
                'adaptation_kwargs': adaptation_kwargs,
                'layer_kwargs': layer_kwargs,
                't_steps': t_steps,
                'd_fc': 100,
                'batch_size': 128,
                'lr': 1e-3,
            },
                name=f'num_maps_cifar10_30ep_{str(adaptation_module)}_n_feature={n_featuremap}')

            trainer = pl.Trainer(max_epochs=num_epoch, logger=wandb_logger)
            hooked_model.cuda()
            # test_results = trainer.test(model, dataloaders=test_loader)
            wandb_logger.watch(hooked_model, log='all', log_freq=1000)

            trainer.fit(model, train_loader, )

            # test
            test_results = trainer.test(model, dataloaders=test_loader)
            logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': repeat_noise,
                                'test_acc': test_results[0]["test_acc"]})
            logger.log_metrics(
                {'num_maps': n_featuremap, 'adaptation_type': f'{str(adaptation_module)}_is_baseline={no_time}'} |
                test_results[0]
            )
            logger.save()
            trainer.save_checkpoint(
                f'learned_models/num_maps_cifar10_30ep_is_baseline={no_time}_{str(adaptation_module)}_num_maps={n_featuremap}.ckpt')

            print('checkpoint saved')
