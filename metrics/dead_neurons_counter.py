from typing import Union, Literal

import torch
from torchmetrics import Metric
from transformer_lens import ActivationCache

from HookedRecursiveCNN import HookedRecursiveCNN


class DeadNeuronMetric(Metric):
    """
        DeadNeuronMetric is a custom metric for tracking the percentage of dead neurons or dead feature maps
        in a neural network during training, specifically designed for use with a HookedRecursiveCNN model.

        Dead neurons are defined as neurons that have zero or near-zero activation across all examples in the dataset
        and across all timesteps in a recurrent layer. This metric helps in identifying layers in the network
        where neurons are not activating and might not be contributing to the model's learning, potentially
        indicating issues such as exploding gradients or improper initialization.

        The class allows the calculation of dead neurons or dead feature maps (depending on the 'per' argument)
        and supports distributed data parallel training.

        Attributes:
            n_adapt_layers (int): Number of adaptive layers in the HookedRecursiveCNN model (will be probed
                                    after the adaptation layer).
            n_timesteps (int): Number of timesteps in the recurrent layers.
            per (Literal['neuron', 'map']): If 'neuron', the metric calculates the percentage of dead neurons.
                                             If 'map', the metric calculates the percentage of dead feature maps.
            dist_sync_on_step (bool): If True, synchronizes the state across devices/gpus on each forward step.
                                      If False, synchronization will be deferred to the end of the epoch.

        Methods:
            update(model: HookedRecursiveCNN, batch): Update the state of the metric based on the
                                                        HookedRecursiveCNN model and the input batch.
            compute(): Compute and return the final dead neuron/feature map percentage for each adaptive layer.
        """
    def __init__(self, n_adapt_layers, n_timesteps, per: Literal['neuron', 'map'] = 'neuron', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_adapt_layers = n_adapt_layers
        self.n_timesteps = n_timesteps
        self.reduce_dims = 0 if per == 'neuron' else (0, 2, 3)
        self.per = per

        for adapt_layer in range(n_adapt_layers):
            self.add_state(f"dead_neurons_{adapt_layer}", default=[], dist_reduce_fx="sum")
            # add_state makes sure variables are properly synced across devices

    def update(self, cache: ActivationCache):
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
