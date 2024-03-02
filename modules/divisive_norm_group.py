import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DivisiveNormGroup(nn.Module):
    def __init__(self, epsilon, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma, sqrt=False ):
        super().__init__()

        self.epsilon = epsilon  # 1.e-8

        self.K = nn.Parameter(torch.tensor(K_init, dtype=torch.float32), requires_grad=train_K)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=train_alpha)
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32), requires_grad=train_sigma)

        self.sqrt = sqrt

    def get_init_actvs(self, x, num_layer):
        """ Get the initial activations. """
        init_actvs = [torch.zeros_like(x, requires_grad=False)]
        return init_actvs

    def params(self):
        return {
            'K': self.K,
            'alpha': self.alpha,
            'sigma': self.sigma,
        }

    def sigmoid_scaled_shifted(self, x, max_val=None):
        if max_val is None:
            max_val = 1
        return max_val * torch.sigmoid(x - (max_val / 2))

    def forward(self, x, G_prev):
        # Initial linear response
        L = torch.relu(x)

        # Rescale feedback signal if it exceeds maximal attainable response
        G_prev = torch.clamp(G_prev, min=torch.zeros_like(G_prev), max=torch.ones_like(G_prev) * self.K)

        if self.sqrt:
            F = torch.sqrt(self.K - G_prev + self.epsilon) / (self.sigma + self.epsilon)
        else:
            F = (self.K - G_prev + self.epsilon) / (self.sigma + self.epsilon)

        # Compute the final response
        response = torch.relu(L * F)

        # Pad the response for surrounding unit calculations
        response_padded = torch.nn.functional.pad(response, (1, 1, 1, 1))

        # Compute total activity of surrounding units
        height, width = response.shape[2:]
        surrounding_activity_sum = sum([
            response_padded[:, :, i:height + 1 + i, j:width + 1 + j]
            for i in [-1, 0, 1] for j in [-1, 0, 1]
        ])

        # Update the feedback signal
        alpha = self.sigmoid_scaled_shifted(self.alpha)
        G = (1 - alpha) * G_prev + alpha * surrounding_activity_sum

        return response, G