from functools import partial

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
from utils.transforms import Identity, RandomRepeatedNoise, Grey, MeanFlat
from utils.visualization import visualize_first_batch_with_timesteps

if __name__ == '__main__':

    num_epochs = [1, 2]
    dataset = 'fashion_mnist'

    # Define transforms for data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    logger = CSVLogger('experimental_data', name='additive_adaptation')

    for num_epoch in num_epochs:
        for contrast  in [0.8, 1.0, 0.2, 0.4, 0.6,]:
            for repeat_noise in [True, False]:
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

                cifar_architecture = True if dataset == 'cifar10' else False
                model = Adaptation(10, cifar_architecture=cifar_architecture)
                trainer = pl.Trainer(max_epochs=num_epoch)
                trainer.fit(model, train_loader, test_loader, )

                # test
                test_results = trainer.test(model, dataloaders=test_loader)
                print(f'Contrast {contrast}, repeat_noise {repeat_noise}: {test_results[0]["test_acc"]}')
                logger.log_metrics({'contrast': contrast, 'epoch': num_epoch, 'repeat_noise': repeat_noise, 'test_acc': test_results[0]["test_acc"]})
                logger.save()

    print('stop')
