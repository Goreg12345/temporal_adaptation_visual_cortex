import numpy as np
import torchvision
from matplotlib import pyplot as plt


def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), vmin=0, vmax=1)
    plt.show()

def visualize_first_batch(dataloader):
    # Get the first batch of images and labels
    images, labels = next(iter(dataloader))

    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images)

    # Show the images
    imshow(img_grid)

    # Print the labels
    print(' '.join('%5s' % labels[j].item() for j in range(4)))

def visualize_first_batch_with_timesteps(dataloader, num_rows=8):
    # Get the first batch of images and labels
    images, labels = next(iter(dataloader))

    # Calculate the number of rows and columns for the plot
    num_rows = min(len(images), num_rows)
    num_cols = images.shape[1]

    # Create a subplot for each image and its timesteps
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.reshape(num_rows, num_cols)

    for i in range(num_rows):
        for j in range(num_cols):
            img = images[i, j] / 2 + 0.5  # Unnormalize
            npimg = img.numpy()
            axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)), vmin=0, vmax=1, cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()