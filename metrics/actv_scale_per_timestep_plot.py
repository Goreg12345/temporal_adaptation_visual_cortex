from typing import Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torchmetrics import Metric
from transformer_lens import ActivationCache



class ActvScalePerTimestepPlot(Metric):
    def __init__(self, repeated_noise: List[bool], n_layers: int, n_timesteps):
        super().__init__()

        self.repeated_noise = repeated_noise
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps

        for timestep in range(self.n_timesteps):
            for noise in repeated_noise:
                for layer in range(self.n_layers):
                    self.add_state(f'cumsum_{noise}_{layer}_{timestep}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)
                    self.add_state(f'total_{noise}_{layer}_{timestep}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)

    def update(self, cache: ActivationCache, repeated_noises):
        # logits: (batch_size, num_classes)
        for timestep in range(self.n_timesteps):
            for noise in repeated_noises.unique():
                for layer in range(self.n_layers):
                    adapt_actvs = cache[f'hks.adapt_{layer}_{timestep}'].sum()
                    n_units = cache[f'hks.adapt_{layer}_{timestep}'].numel()

                    cumsum_state = getattr(self, f'cumsum_{noise}_{layer}_{timestep}')
                    total_state = getattr(self, f'total_{noise}_{layer}_{timestep}')

                    cumsum_state += adapt_actvs
                    total_state += n_units

                    setattr(self, f'cumsum_{noise}_{layer}_{timestep}', cumsum_state)
                    setattr(self, f'total_{noise}_{layer}_{timestep}', total_state)

    def compute(self):
        results = {
            'mean_actv': [],
            'timestep': [],
            'noise': [],
            'layer': []
        }
        for timestep in range(self.n_timesteps):
            for layer in range(self.n_layers):
                for noise in self.repeated_noise:
                    cumsum = getattr(self, f'cumsum_{noise}_{layer}_{timestep}')
                    total = getattr(self, f'total_{noise}_{layer}_{timestep}')

                    results['mean_actv'].append((cumsum / total).detach().cpu().numpy())
                    results['timestep'].append(timestep)
                    results['noise'].append(noise)
                    results['layer'].append(layer)
        results = pd.DataFrame(results)
        results['mean_actv'] = np.stack(results['mean_actv'])
        results['layer'] = results['layer'].astype(str)
        # sns.relplot(data=results, x='timestep', y='mean_actv', hue='noise', col='layer', kind='line', facet_kws={'sharey': False})
        sns.lineplot(data=results, x='timestep', y='mean_actv', hue='noise', style='layer', )
        return plt  # wandb will fetch the figure from plt and log it
