from typing import List

import torch
from torchmetrics import Metric


class AccuracyPerSetting(Metric):
    def __init__(self, contrasts: List[float], repeated_noise: List[bool]):
        super().__init__()

        self.contrasts = contrasts
        self.repeated_noise = repeated_noise

        for contrast in contrasts:
            for noise in repeated_noise:
                self.add_state(f'correct_{contrast}_{noise}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)
                self.add_state(f'total_{contrast}_{noise}', default=torch.tensor(0.), dist_reduce_fx=torch.sum)

    def update(self, logits, targets, contrasts, repeated_noises):
        # logits: (batch_size, num_classes)
        for contrast in contrasts.unique():
            for noise in repeated_noises.unique():
                l = logits[(torch.isclose(contrasts, contrast) & (repeated_noises == noise)).flatten()]
                t = targets[(torch.isclose(contrasts, contrast) & (repeated_noises == noise)).flatten()]

                c = "{:.1f}".format(contrast.item())
                correct_state = getattr(self, f'correct_{c}_{noise}')
                total_state = getattr(self, f'total_{c}_{noise}')

                correct_state += (torch.argmax(l, dim=1) == t).sum()
                total_state += len(t)

                setattr(self, f'correct_{c}_{noise}', correct_state)
                setattr(self, f'total_{c}_{noise}', total_state)

    def compute(self):
        results = {}
        for contrast in self.contrasts:
            for noise in self.repeated_noise:
                correct_state = getattr(self, f'correct_{contrast}_{noise}')
                total_state = getattr(self, f'total_{contrast}_{noise}')

                results[f'acc/{contrast}_{noise}'] = correct_state / total_state
        return results

