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

class RandomRepeatedNoise:
    """
    Randomly adds noise to the image but the same noise is added to all timesteps
    """
    def __init__(self, contrast=1, repeat_noise_at_test=True):
        self.noise_seed = dict()
        self.contrast = contrast
        self.repeat_noise_at_test = repeat_noise_at_test
        self.noise_seed_test = dict()
    def __call__(self, x, index, _, stage='adapter'):
        if stage == 'adapter':
            if index in self.noise_seed:
                rng_state = torch.get_rng_state()
                torch.manual_seed(self.noise_seed[index])
                noise = torch.rand_like(x)
            else:
                # generate randint
                self.noise_seed[index] = torch.randint(0, 2**32, (1,)).item()
                rng_state = torch.get_rng_state()
                torch.manual_seed(self.noise_seed[index])
                noise = torch.rand_like(x)
            torch.set_rng_state(rng_state)
            return noise
        if stage == 'test':
            if self.repeat_noise_at_test:
                rng_state = torch.get_rng_state()
                torch.manual_seed(self.noise_seed[index])
                noise = torch.rand_like(x)
            else:
                if index in self.noise_seed_test:
                    rng_state = torch.get_rng_state()
                    torch.manual_seed(self.noise_seed_test[index])
                    noise = torch.rand_like(x)
                else:
                    # generate randint
                    self.noise_seed_test[index] = torch.randint(0, 2**32, (1,)).item()
                    rng_state = torch.get_rng_state()
                    torch.manual_seed(self.noise_seed_test[index])
                    noise = torch.rand_like(x)
            torch.set_rng_state(rng_state)
            return noise - .5 + self.contrast * x

