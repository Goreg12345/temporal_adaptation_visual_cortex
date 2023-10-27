from typing import Union, Literal

import torch
from torchmetrics import Metric

from HookedRecursiveCNN import HookedRecursiveCNN


class DeadNeuronMetric(Metric):
    def __init__(self, n_adapt_layers, n_timesteps, per: Literal['neuron', 'map'] = 'neuron', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_adapt_layers = n_adapt_layers
        self.n_timesteps = n_timesteps
        self.reduce_dims = 0 if per == 'neuron' else (0, 2, 3)
        self.per = per

        for adapt_layer in range(n_adapt_layers):
            self.add_state(f"dead_neurons_{adapt_layer}", default=[], dist_reduce_fx="sum")
            # add_state makes sure variables are properly synced across devices

    def update(self, model: HookedRecursiveCNN, batch):
        # Update counts
        X, _ = batch
        _, cache = model.run_with_cache(X)
        for layer in range(self.n_adapt_layers):
            dead_neurons_state = getattr(self, f'dead_neurons_{layer}')

            if len(dead_neurons_state) == 0:  # INIT
                dead_neurons_state = torch.zeros_like(cache[f'hks.adapt_{layer}_0'].sum(dim=self.reduce_dims))

            for t in range(self.n_timesteps):
                dead_neurons_state += cache[f'hks.adapt_{layer}_{t}'].sum(dim=self.reduce_dims)
                # batch channels width height -> sum over batch -> channels width height
            setattr(self, f'dead_neurons_{layer}', dead_neurons_state)

    def compute(self):
        results = {}
        for layer in range(self.n_adapt_layers):
            dead_neurons_state = getattr(self, f'dead_neurons_{layer}')
            dead_neurons_percent = torch.sum(dead_neurons_state <= 0 + 1e-8) / dead_neurons_state.numel()
            results[f'dead_{self.per}_percent/{layer}'] = dead_neurons_percent
        return results
