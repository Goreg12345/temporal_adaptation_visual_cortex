import torch


class Identity:
    def __call__(self, x, idx, target):
        return x

class Zeros:
    def __call__(self, x, idx, target):
        return torch.zeros_like(x)

class Grey:
    def __call__(self, x, idx, target):
        return torch.ones_like(x) * 0.5


class MeanFlat:
    def __call__(self, x, idx, target):
        # mean per channel
        return torch.ones_like(x) * torch.mean(x, dim=(1, 2), keepdim=True)

class RandomRepeatedNoise:
    """
    Randomly adds noise to the image but the same noise is added to all timesteps
    """
    def __init__(self, contrast=1, repeat_noise_at_test=True, first_in_line=False):
        self.noise_seed = dict()
        self.contrast = contrast
        self.repeat_noise_at_test = repeat_noise_at_test
        self.noise_seed_test = dict()

    def __call__(self, x, index, _, stage='adapter', first_in_line=False):
        if stage == 'adapter':
            if index in self.noise_seed:
                rng_state = torch.get_rng_state()
                torch.manual_seed(self.noise_seed[index])
            else:
                # generate randint
                self.noise_seed[index] = torch.randint(0, 2**32, (1,)).item()
                rng_state = torch.get_rng_state()
                torch.manual_seed(self.noise_seed[index])
            noise = torch.rand_like(x) -.5
            torch.set_rng_state(rng_state)
            return (noise + torch.mean(x, dim=(1, 2), keepdim=True)).clamp(0,1)
        if stage == 'test':
            if self.repeat_noise_at_test:
                rng_state = torch.get_rng_state()
                torch.manual_seed(self.noise_seed[index])
            else:
                if not first_in_line:  # use noise from previous image
                    if index not in self.noise_seed_test:
                        raise ValueError('this RandomNoiseRepeater is not first in line but no noise was generated '
                                         'for the previous image. Call a Noise transform that is first in line first.')

                    rng_state = torch.get_rng_state()
                    torch.manual_seed(self.noise_seed_test[index])
                else:  # if first in line, generate new noise
                    # generate randint
                    self.noise_seed_test[index] = torch.randint(0, 2**32, (1,)).item()
                    rng_state = torch.get_rng_state()
                    torch.manual_seed(self.noise_seed_test[index])
            noise = torch.rand_like(x) -.5
            torch.set_rng_state(rng_state)

            mean = torch.mean(x)
            # Apply contrast adjustment
            adjusted_image = (x - mean) * self.contrast + mean
            # mixing_ratio = 0.7
            # noised_image = x * (1 - mixing_ratio) + noise * mixing_ratio
            # return noised_image
            return (noise + adjusted_image).clamp(0,1)

