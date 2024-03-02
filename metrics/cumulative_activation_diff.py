from typing import List

import torch
from torchmetrics import Metric
from transformer_lens import ActivationCache


class CumulativeActivationDiff(Metric):
    def __init__(self, contrasts: List[float], repeated_noise: List[bool], n_layers: int):
        super().__init__()

        self.contrasts = contrasts
        self.repeated_noise = repeated_noise
        self.n_layers = n_layers

        for contrast in contrasts:
            for noise in repeated_noise:
                for layer in range(self.n_layers):
                    self.add_state(f'cumsum_{contrast}_{noise}_{layer}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)
                    self.add_state(f'total_{contrast}_{noise}_{layer}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)

    def update(self, cache: ActivationCache, contrasts, repeated_noises):
        # logits: (batch_size, num_classes)
        for contrast in contrasts.unique():
            for noise in repeated_noises.unique():
                for layer in range(self.n_layers):
                    last_timestep = max([int(key.split('_')[-1]) for key in cache.keys() if f'adapt_{layer}' in key])
                    adapt_actvs = cache[f'hks.adapt_{layer}_{last_timestep}'].sum()
                    n_units = cache[f'hks.adapt_{layer}_{last_timestep}'].numel()

                    c = "{:.1f}".format(contrast.item())
                    cumsum_state = getattr(self, f'cumsum_{c}_{noise}_{layer}')
                    total_state = getattr(self, f'total_{c}_{noise}_{layer}')

                    cumsum_state += adapt_actvs
                    total_state += n_units

                    setattr(self, f'cumsum_{c}_{noise}_{layer}', cumsum_state)
                    setattr(self, f'total_{c}_{noise}_{layer}', total_state)

    def compute(self):
        results = {}
        for contrast in self.contrasts:
            for layer in range(self.n_layers):
                rep_noise_state = getattr(self, f'cumsum_{contrast}_{True}_{layer}')
                new_noise_state = getattr(self, f'cumsum_{contrast}_{False}_{layer}')
                total_state = getattr(self, f'total_{contrast}_{True}_{layer}')

                results[f'cum_actv_diff/{contrast}_l{layer}'] = (new_noise_state - rep_noise_state) / total_state
        return results

