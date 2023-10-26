from typing import Dict

import torch
from torchmetrics import Metric


class RepeatedToNovelAccuracy(Metric):
    """
    For every contrast level, calculate the accuracy of repeated noise images minus the accuracy of novel noise images
    """
    def __init__(self, datasets: Dict, dist_sync_on_step=False):
        """
        :param datasets:
        :param dist_sync_on_step:
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n_correct_repeated", default=[], dist_reduce_fx=None)
        self.add_state("n_total_repeated", default=[], dist_reduce_fx=None)
        self.add_state("n_correct_novel", default=[], dist_reduce_fx=None)
        self.add_state("n_total_novel", default=[], dist_reduce_fx=None)

    def update(self, params):
        current_params = {name: p.clone().detach() for name, p in params}

        # If this is the first call, just save the current parameters
        if not self.param_updates:
            self.param_updates.append(current_params)
            self.layer_names.extend(list(current_params.keys()))
            return

        # Compute parameter differences and their norms
        param_diffs = {}
        for name, p in current_params.items():
            prev_p = self.param_updates[-1][name]
            # Calculate the norm of the update relative to the norm of the parameter
            relative_norm = torch.norm(p - prev_p) / (torch.norm(prev_p) + 1e-6)  # Prevent division by zero
            param_diffs[name] = relative_norm

        # Save the norms and the current parameters for the next round
        self.param_updates[-1] = param_diffs
        self.param_updates.append(current_params)

    def compute(self):
        # Skip the first entry since it's the initial parameters, not an update
        updates = self.param_updates[:-1]  # exclude the last since it's the final parameters, not an update

        if not updates:
            return {}

        # Compute average update norms per layer
        avg_update_norms = {}
        for name in self.layer_names:
            if type(name) == Dict:
                print('stop')
            norms = [update[name] for update in updates]
            avg_update_norms[name] = torch.stack(norms).mean().item()

        self.param_updates = self.param_updates[-1:]

        return avg_update_norms
