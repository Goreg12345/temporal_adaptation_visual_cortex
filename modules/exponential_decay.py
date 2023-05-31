import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ExponentialDecay(nn.Module):
    def __init__(self, alpha_init, train_alpha, beta_init, train_beta):
        super().__init__()

        self.alpha = nn.Parameter(alpha_init, requires_grad=train_alpha)
        self.beta = nn.Parameter(beta_init, requires_grad=train_beta)
        self.relu = nn.ReLU()

    def get_init_actvs(self, x, num_layer):
        """ Get the initial activations. """
        init_actvs = (torch.zeros_like(x, requires_grad=False), torch.zeros_like(x, requires_grad=False))
        return init_actvs

    def forward(self, x, previous_input, previous_state):
        """
        Perform forward pass.

        Args:
            previous_input (torch.Tensor): Input from previous step.
            previous_state (torch.Tensor): State from previous step.

        Returns:
            updated_state (torch.Tensor): Updated state.
            state_beta_update (torch.Tensor): Updated state with beta applied.
        """

        # Calculate the updated state using a weighted sum of the previous state and input
        updated_state = previous_state * self.alpha + previous_input * (1 - self.alpha)

        # Apply beta to the updated state
        state_beta_update = self.beta * updated_state

        new_actv = self.relu(x - state_beta_update)

        x = new_actv.clone()

        return x, new_actv, updated_state
