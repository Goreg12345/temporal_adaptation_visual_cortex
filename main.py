from functools import partial

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch


from models.noisy_dataloader import FashionMNISTNoisyDataset
from utils.transforms import Identity, RandomRepeatedNoise, Grey
from utils.visualization import visualize_first_batch_with_timesteps

if __name__ == '__main__':
    # Define transforms for data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    noise_transformer = RandomRepeatedNoise(contrast=1, repeat_noise_at_test=False)
    noise_transformer_test = partial(noise_transformer, stage='test')

    timestep_transforms = [Identity()] + [Grey()] + [noise_transformer] * 5 + [Grey()] + [noise_transformer_test] * 3
    # Create instances of the Fashion MNIST dataset
    train_dataset = FashionMNISTNoisyDataset('train', transform, timestep_transforms)
    #test_dataset = FashionMNISTNoisyDataset('test', transform, timestep_transforms)

    # Create the DataLoaders for the Fashion MNIST dataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    #test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Print the length of the DataLoader
    print(f'Train DataLoader: {len(train_loader)} batches')
    #print(f'Test DataLoader: {len(test_loader)} batches')

    # Visualize the first batch of images
    visualize_first_batch_with_timesteps(train_loader, 8)

    print('stop')
