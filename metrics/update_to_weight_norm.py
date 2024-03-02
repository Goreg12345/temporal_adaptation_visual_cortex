from typing import Dict

import torch
from torchmetrics import Metric


class RelativeGradientUpdateNorm(Metric):
    """
        RelativeGradientUpdateNorm is a Metric for monitoring the training progress of a neural network
        by calculating the relative norm of parameter updates during training. This metric helps in identifying
        issues related to the training dynamics, such as vanishing or exploding gradients.

        The relative norm of parameter updates is defined as the Euclidean norm of the difference between the
        current and previous parameter values, divided by the Euclidean norm of the previous parameter values.
        In other words, it is the norm of the parameter update relative to the norm of the parameter.
        This provides a scale-invariant measure of how much the parameters are changing at each training step,
        which can be crucial for understanding and diagnosing training issues.

        Typically, these parameters should be around 1e-3. If they are off, consider changing the learning rate,

        It needs that parameters of the model, e.g. model.named_parameters().

        Attributes:
            params (List[torch.nn.Parameter]): A list of PyTorch parameters from the model being trained.
            dist_sync_on_step (bool): If True, synchronizes the metric state across devices/gpus on each step.
                                      If False, synchronization will be deferred to the end of the epoch.

        Methods:
            __init__(self, params, dist_sync_on_step=False): Initializes the RelativeGradientUpdateNorm metric.
            update(self, params): Updates the metric state based on the current values of the parameters.
            compute(self): Computes and returns the mean relative norm of parameter updates for each parameter.

        The metric state includes the previous values of the parameters, the accumulated norms of the parameter
        updates, and a count of the number of updates. These are used to compute the mean relative norm of the
        parameter updates when the `compute` method is called.
        """
    def __init__(self, params, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        for name, p in params:
            self.add_state(name, default=p.clone().detach(), dist_reduce_fx=None)

    def update(self, params):
        # Compute parameter differences and their norms
        for name, p in params:
            p = p.clone().detach()
            prev_ps = getattr(self, name)
            if type(prev_ps) == torch.Tensor:
                prev_ps = [prev_ps]
            prev_p = prev_ps[-1]
            # Calculate the norm of the update relative to the norm of the parameter
            relative_norm = torch.norm(p - prev_p) / (torch.norm(prev_p) + 1e-6)  # Prevent division by zero
            prev_ps[-1] = relative_norm
            prev_ps.append(p)
            setattr(self, name, prev_ps)
            self._defaults[name] = prev_ps

    def compute(self):
        updates = {}
        for name, p in self._defaults.items():
            # Skip the first entry since it's the initial parameters, not an update
            if not p[:-1]:
                continue
            updates[name] = torch.stack(p[:-1]).mean().item()
            self._defaults[name] = [p[-1]]
            setattr(self, name, [p[-1]])
        return updates
