import torch
from torchmetrics import Metric
from transformer_lens import ActivationCache


class Sparsity(Metric):
    """
        Sparsity Metric to calculate the sparsity of activations in a HookedRecursiveCNN model.

        The sparsity is calculated as the ratio of zero-valued activations to the total number of activations
        in specified layers and timesteps of the model.

        Attributes:
            n_adapt_layers (int): The number of adaptive layers in the HookedRecursiveCNN model for which
                                  sparsity will be calculated.
            timestep (int): The specific timestep at which to calculate the sparsity. If set to a negative
                            value, the timestep is counted from the last timestep backwards (e.g., -1 for the
                            last timestep). Default is -1.

        Methods:
            __init__(self, n_adapt_layers, timestep=-1): Initializes the Sparsity metric.
            update(self, model, batch): Updates the metric's state using the output from the HookedRecursiveCNN model.
            compute(self): Computes and returns the sparsity for each layer as a dictionary.

        The update method expects a batch of input data and a HookedRecursiveCNN model as input. It uses the
        model's run_with_cache method to get the activations at the specified timestep and calculates the
        number of zero-valued activations and the total number of activations. The compute method then calculates
        and returns the sparsity for each layer.
        """
    def __init__(self, n_adapt_layers: int, timestep: int = -1):
        super().__init__()

        self.n_adapt_layers = n_adapt_layers
        self.timestep = timestep

        for layer in range(n_adapt_layers):
            self.add_state(f'zero_neurons_{layer}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)
            self.add_state(f'total_neurons_{layer}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)

    def update(self, cache: ActivationCache):
        if self.timestep < 0:
            # get maximum timestep
            max_timestep = max([int(key.split('_')[-1]) for key in cache.keys() if 'adapt' in key])
            self.timestep = max_timestep + self.timestep + 1
        for layer in range(self.n_adapt_layers):
            zero_neurons_state = getattr(self, f'zero_neurons_{layer}')
            total_neurons_state = getattr(self, f'total_neurons_{layer}')

            zero_neurons_state += cache[f'hks.adapt_{layer}_{self.timestep}'].eq(0).sum()
            total_neurons_state += cache[f'hks.adapt_{layer}_{self.timestep}'].numel()

            setattr(self, f'zero_neurons_{layer}', zero_neurons_state)
            setattr(self, f'total_neurons_{layer}', total_neurons_state)

    def compute(self):
        results = {}
        for layer in range(self.n_adapt_layers):
            zero_neurons_state = getattr(self, f'zero_neurons_{layer}')
            total_neurons_state = getattr(self, f'total_neurons_{layer}')
            results[f'sparsity/{layer}'] = zero_neurons_state / total_neurons_state
        return results

