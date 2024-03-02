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
    def __init__(self, contrast='random', repeat_noise_at_test=True, first_in_line=False,
                 noise_seed=None, noise_seed_test=None, noise_component='test',
                 noise_type='uniform', noise_offset=0, noise_mean_offset=0, noise_std=1):
        self.noise_seed = noise_seed if noise_seed else dict()
        self.contrast = contrast
        self.repeat_noise_at_test = repeat_noise_at_test
        self.noise_seed_test = noise_seed_test if noise_seed_test else dict()

        self.noise_type = noise_type
        self.noise_offset = noise_offset
        self.noise_mean_offset = noise_mean_offset
        self.noise_std = noise_std

        if not noise_component in ['test', 'adapter', 'both']:
            raise ValueError('noise_component must be one of "test", "adapter", "both"')
        self.noise_component = noise_component

    def noise_like(self, x):
        if self.noise_type == 'uniform':
            return torch.rand_like(x) - .5
        elif self.noise_type == 'normal':
            return torch.randn_like(x) * self.noise_std + self.noise_mean_offset
        else:
            raise ValueError(f'noise_type must be one of "uniform", "normal", not {self.noise_type}')

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
            noise = self.noise_like(x)
            torch.set_rng_state(rng_state)
            noise = (noise + torch.mean(x, dim=(1, 2), keepdim=True)).clamp(0,1)
            if self.noise_component in ['adapter', 'both']:
                noise = (noise - noise.mean()) * self.contrast + noise.mean()
            return noise
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
            noise = self.noise_like(x)
            # shift noise horizontally anvertically by noise offset
            noise = torch.roll(noise, shifts=(self.noise_offset, self.noise_offset), dims=(1, 2))

            mean = torch.mean(x)
            if self.contrast == 'random':
                contrast = torch.rand(1)
            else:
                contrast = self.contrast

            if self.noise_component == 'test':
                adjusted_image = (x - mean) * contrast + mean
            elif self.noise_component == 'adapter':
                adjusted_image = x
                noise = (noise + mean) * contrast - mean
            elif self.noise_component == 'both':
                adjusted_image = (x - mean) * contrast + mean
                noise = (noise + mean) * contrast - mean

            torch.set_rng_state(rng_state)

            # return noised_image
            return (noise + adjusted_image).clamp(0,1)

