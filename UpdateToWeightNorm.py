from typing import Dict

import torch
from torchmetrics import Metric


class RelativeGradientUpdateNorm(Metric):
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
