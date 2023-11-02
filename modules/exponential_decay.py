from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor


class ExponentialDecay(nn.Module):
    def __init__(self, alpha_init, train_alpha, beta_init, train_beta):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=train_alpha)
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32), requires_grad=train_beta)
        self.relu = nn.ReLU()

    def get_init_actvs(self, x, num_layer):
        """ Get the initial activations. """
        init_actvs = (torch.zeros_like(x, requires_grad=False), torch.zeros_like(x, requires_grad=False))
        return init_actvs

    def params(self) -> Dict[str, torch.Tensor]:
        return {
            'alpha': self.alpha,
            'beta': self.beta,
        }

    def adaptation_graph(self, inputs: Tensor) -> Dict[str, Tensor]:
        """ given the current adaptation params and a list of inputs, calculate adaptation output and state given the current adaptation params"""
        outputs = []
        new_actvs = []
        states = []
        inputs = inputs.to(self.alpha.device)
        previous_response, previous_state = self.get_init_actvs(inputs[0], 0)
        for x in inputs:
            x, previous_response, previous_state = self.forward(x, previous_response, previous_state)
            outputs.append(x)
            new_actvs.append(previous_response)
            states.append(previous_state)
        return {
            'response': torch.stack(outputs),
            'new_activations': torch.stack(new_actvs),
            'state': torch.stack(states)
        }

    def forward(self, x, previous_response, previous_state):
        """
        Perform forward pass.

        Args:
            previous_response (torch.Tensor): Input from previous step.
            previous_state (torch.Tensor): State from previous step.

        Returns:
            updated_state (torch.Tensor): Updated state.
            state_beta_update (torch.Tensor): Updated state with beta applied.
        """

        # Calculate the updated state using a weighted sum of the previous state and input
        updated_state = previous_state * self.alpha + previous_response * (1 - self.alpha)

        # Apply beta to the updated state
        state_beta_update = self.beta * updated_state

        new_actv = self.relu(x - state_beta_update)

        x = new_actv.clone()

        return x, new_actv, updated_state
