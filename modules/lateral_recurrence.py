import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class LateralRecurrence(nn.Module):
    def __init__(self, is_conv, n_channels, ):
        super().__init__()

        if is_conv:
            self.lat_rec = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1,)
        else:
            self.lat_rec = nn.Linear(in_features=n_channels, out_features=n_channels,)
        self.relu = nn.ReLU()

    def get_init_actvs(self, x, num_layer):
        """ Get the initial activations. """
        # if shape is 4D, then we are dealing with a convolutional layer
        if len(x.shape) == 4:
            return [self.relu(x).clone()]
        # if shape is 2D, then we are dealing with a fully connected layer
        elif len(x.shape) == 2:
            return [x.clone()]
        # raise error if shape is not 2D or 4D
        raise ValueError("Shape of input is not 2D or 4D.")

    def forward(self, x, previous_input,):
        x_r = self.lat_rec(previous_input)
        new_actv = x + x_r

        # if conv layer, then apply relu
        if len(x.shape) == 4:
            new_actv = self.relu(new_actv)

        x = new_actv.clone()

        return x, new_actv
