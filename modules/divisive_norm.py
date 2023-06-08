import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DivisiveNorm(nn.Module):
    def __init__(self, epsilon, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma ):
        super().__init__()

        self.epsilon = epsilon  # 1.e-8

        self.K = nn.Parameter(torch.tensor(K_init, dtype=float), requires_grad=train_K)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=float), requires_grad=train_alpha)
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=float), requires_grad=train_sigma)

    def get_init_actvs(self, x, num_layer):
        """ Get the initial activations. """
        init_actvs = (torch.zeros_like(x, requires_grad=False), torch.zeros_like(x, requires_grad=False))
        return init_actvs

    def forward(self, x, F_prev, G_prev):
        L = torch.relu(x)  # linear response

        if G_prev.max() > self.K:                                                  # rescale if feedback signal exceeds maximal attainable response
            G_prev = G_prev / G_prev.max()
            G_prev = G_prev * self.K

        F = torch.sqrt(self.K - G_prev + self.epsilon) / self.sigma  # multiplicative feedback
        x = torch.relu(L * F)  # response

        # compute G by adding activity of unit
        G = ((1 - self.alpha) * G_prev) + self.alpha * x  # update feedback signal

        return x, F, G