import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DivisiveNormGroup(nn.Module):
    def __init__(self, epsilon, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma ):
        super().__init__()

        self.epsilon = epsilon  # 1.e-8

        self.K = nn.Parameter(torch.tensor(K_init, dtype=float), requires_grad=train_K)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=float), requires_grad=train_alpha)
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=float), requires_grad=train_sigma)

    def get_init_actvs(self, x, num_layer):
        """ Get the initial activations. """
        init_actvs = [torch.zeros_like(x, requires_grad=False)]
        return init_actvs

    def forward(self, x, G_previous):
        # Initial linear response
        linear_response = torch.relu(x)

        # Rescale feedback signal if it exceeds maximal attainable response
        if torch.max(G_previous) > self.K:
            # Rescale the feedback signal to range [0,1]
            G_previous /= torch.max(G_previous)

            # Scale to range [0,K]
            G_previous *= self.K

            # Compute the difference between max attainable response and the feedback signal
        difference = torch.subtract(self.K, G_previous)

        # Add epsilon for stability, then take square root
        modified_difference = torch.sqrt(difference + self.epsilon)

        # Compute the multiplicative feedback
        multiplicative_feedback = modified_difference / self.sigma

        # Compute the final response
        response = torch.relu(torch.mul(linear_response, multiplicative_feedback))

        # Pad the response for surrounding unit calculations
        response_padded = torch.nn.functional.pad(response, (1, 1, 1, 1))

        # Compute total activity of surrounding units
        surrounding_activity_sum = sum([
            response_padded[:, :, i:self.height + 1 + i, j:self.width + 1 + j]
            for i in [-1, 0, 1] for j in [-1, 0, 1]
        ])

        # Update the feedback signal
        G = torch.add((1 - self.alpha) * G_previous, self.alpha * surrounding_activity_sum)

        return response, G