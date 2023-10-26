from typing import Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import repeat
from fancy_einsum import einsum


class DivisiveNormChannel(nn.Module):
    def __init__(self, n_channels, epsilon, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma):
        super().__init__()

        self.epsilon = epsilon

        self.K = nn.Parameter(torch.ones((n_channels,), dtype=torch.float32) * K_init, requires_grad=train_K)
        self.alpha = nn.Parameter(torch.ones((n_channels,), dtype=torch.float32) * alpha_init, requires_grad=train_alpha)
        self.sigma = nn.Parameter(torch.ones((n_channels,), dtype=torch.float32) * sigma_init, requires_grad=train_sigma)

        self.cur_layer = 0

        # clamp alpha between 0 and 1?
    def get_init_actvs(self, x, num_layer):
        return [torch.zeros_like(x, requires_grad=False)]

    def params(self) -> Dict[str, torch.Tensor]:
        return {
            'K': self.K,
            'alpha': self.sigmoid_scaled_shifted(self.alpha),
            'sigma': self.sigma,
        }

    def sigmoid_scaled_shifted(self, x, max_val=None):
        if max_val is None:
            max_val = 1
        return max_val * torch.sigmoid(x - (max_val / 2))

    def forward(self, x, G_prev):
#        L = torch.relu(x)
#
#        self.K.data = torch.clamp(self.K.data, min=self.epsilon)  # make sure K is always bigger than epsilon
#        self.sigma.data = torch.clamp(self.sigma.data, min=self.epsilon)  # make sure sigma is always bigger than epsilon
#        # check if any parameter is nan
#        if torch.isnan(self.K).any() or torch.isnan(self.alpha).any() or torch.isnan(self.sigma).any():
#            print('nan in params')
#
#        K = repeat(self.K, 'channels -> batch channels height width', batch=x.shape[0], height=x.shape[2], width=x.shape[3])
#        sigma = repeat(self.sigma, 'channels -> batch channels height width', batch=x.shape[0], height=x.shape[2], width=x.shape[3])
#        alpha = repeat(self.alpha, 'channels -> batch channels height width', batch=x.shape[0], height=x.shape[2], width=x.shape[3])
#
#        G_prev = self.sigmoid_scaled_shifted(G_prev, K)  # make sure G_prev is between 0 and K without conditionals
#
#        F = torch.sqrt(K - G_prev + self.epsilon) / (sigma + self.epsilon)
#
#        response = torch.relu(L * F)
#
#        G = ((1 - self.sigmoid_scaled_shifted(alpha)) * G_prev) + self.sigmoid_scaled_shifted(alpha) * response
#
#        # check if any response, F or G is nan
#        if torch.isnan(response).any() or torch.isnan(F).any() or torch.isnan(G).any():
#            print('nan in response, F or G')
#        return response, G
        L = torch.relu(x)

        self.K.data = torch.clamp(self.K.data, min=self.epsilon)  # make sure K is always bigger than epsilon
        self.sigma.data = torch.clamp(self.sigma.data, min=self.epsilon)  # make sure sigma is always bigger than epsilon

        K = self.K[self.cur_layer]
        sigma = self.sigma[self.cur_layer]
        alpha = self.alpha[self.cur_layer]
        if self.cur_layer < 2:
            self.cur_layer += 1
        else:
            self.cur_layer = 0



        # check if any parameter is nan
        if torch.isnan(K).any() or torch.isnan(self.alpha).any() or torch.isnan(self.sigma).any():
            print('nan in params')
        #G_prev = self.sigmoid_scaled_shifted(G_prev, K)  # make sure G_prev is between 0 and K without conditionals
        G_prev = torch.clamp(G_prev, min=torch.zeros_like(G_prev), max=torch.ones_like(G_prev) * K)
        # F = torch.sqrt(self.K - G_prev + self.epsilon) / self.
        F = (K - G_prev + self.epsilon) / (sigma + self.epsilon)
        response = torch.relu(L * F)

        G = ((1 - self.sigmoid_scaled_shifted(alpha)) * G_prev) + self.sigmoid_scaled_shifted(alpha) * response
        return response, G
