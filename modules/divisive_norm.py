import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DivisiveNorm(nn.Module):
    def __init__(self, epsilon, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma):
        super().__init__()

        self.epsilon = epsilon

        self.K = nn.Parameter(torch.tensor(K_init, dtype=torch.float32), requires_grad=train_K)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=train_alpha)
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32), requires_grad=train_sigma)

    def get_init_actvs(self, x, num_layer):
        return [torch.zeros_like(x, requires_grad=False)]

    def forward(self, x, G_prev):
        L = torch.relu(x)

        G_max = G_prev.max()
        if G_max > self.K:
            G_prev = (G_prev / G_max) * self.K

        F = torch.sqrt(self.K - G_prev + self.epsilon) / self.sigma
        response = torch.relu(L * F)

        G = ((1 - self.alpha) * G_prev) + self.alpha * response
        return response, G
