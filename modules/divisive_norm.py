from typing import Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor


class DivisiveNorm(nn.Module):
    def __init__(self, epsilon, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma, sqrt=False):
        super().__init__()

        self.epsilon = epsilon

        self.K = nn.Parameter(torch.tensor(K_init, dtype=torch.float32), requires_grad=train_K)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=train_alpha)
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32), requires_grad=train_sigma)

        self.sqrt=sqrt

        # clamp alpha between 0 and 1?
    def get_init_actvs(self, x, num_layer):
        return [torch.zeros_like(x, requires_grad=False)]

    def params(self) -> Dict[str, torch.Tensor]:
        return {
            'K': self.K,
            'alpha': self.sigmoid_scaled_shifted(self.alpha),
            'sigma': self.sigma,
        }

    def adaptation_graph(self, inputs: Tensor) -> Dict[str, Tensor]:
        """ given the current adaptation params and a list of inputs, calculate adaptation output and state given the current adaptation params"""
        outputs = []
        gs = []
        inputs = inputs.to(self.alpha.device)
        g = self.get_init_actvs(inputs[0], 0)[0]
        for x in inputs:
            x, g = self.forward(x, g)
            outputs.append(x)
            gs.append(g)
        return {
            'response': torch.stack(outputs),
            'G': torch.stack(gs)
        }

    def sigmoid_scaled_shifted(self, x, max_val=None):
        if max_val is None:
            max_val = 1
        return max_val * torch.sigmoid(x - (max_val / 2))

    def forward(self, x, G_prev):
        L = torch.relu(x)

        # G_max = G_prev.max()
        # if G_max > self.K:
        #     G_prev = (G_prev / G_max) * self.K
        # if G_prev.min() < 0:
        #     print('G_prev min < 0')
        # G_prev = self.sigmoid_scaled_shifted(G_prev, self.K)  # make sure G_prev is between 0 and K without conditionals
        G_prev = torch.clamp(G_prev, min=torch.zeros_like(G_prev), max=torch.ones_like(G_prev) * self.K)


        # F = torch.sqrt(self.K - G_prev + self.epsilon) / self.
        if self.sqrt:
            F = torch.sqrt(self.K - G_prev + self.epsilon) / (self.sigma + self.epsilon)
        else:
            F = (self.K - G_prev + self.epsilon) / (self.sigma + self.epsilon)
        response = torch.relu(L * F)

        G = ((1 - self.sigmoid_scaled_shifted(self.alpha)) * G_prev) + self.sigmoid_scaled_shifted(self.alpha) * response
        return response, G
