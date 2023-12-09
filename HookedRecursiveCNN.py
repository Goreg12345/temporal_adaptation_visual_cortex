import torch
from torch import nn
from torch.nn import Module
from transformer_lens.hook_points import HookedRootModule, HookPoint


class HookedRecursiveCNN(HookedRootModule):
    def __init__(self, t_steps, layer_kwargs, adaptation_module, adaptation_kwargs, d_fc=1024,
                 decode_every_timestep=False, *args):
        # training variables
        super().__init__(*args)
        self.hks = {}
        for i in range(t_steps):
            for j in range(len(layer_kwargs)):
                self.hks[f'conv_{j}_{i}'] = HookPoint()
                self.hks[f'adapt_{j}_{i}'] = HookPoint()
                self.hks[f'state_{j}_{i}'] = HookPoint()
        self.hks = nn.ModuleDict(self.hks)
        self.t_steps = t_steps

        # activation functions, pooling and dropout layers
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

        self.conv_layers = nn.ModuleList(
            nn.Conv2d(**layer) for layer in layer_kwargs[:-1])  # last is fc layer
        self.adapt_layers = nn.ModuleList(
            adaptation_module(**layer) for layer in adaptation_kwargs[:-1])

        # fc 1
        self.fc1 = nn.Linear(**layer_kwargs[-1])
        self.hook_fc1 = HookPoint()
        # self.adaptfc1 = adaptation_module(**adaptation_kwargs[3])

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=d_fc,
                                 out_features=10)  # only saves the output from the last timestep to train
        self.decode_every_timestep = decode_every_timestep

        self.setup()

    def forward(self, X: torch.Tensor):

        actvs_prev = {}
        if X.shape[1] < 1:
            raise ValueError(f'X must have shape [batch, timesteps, channel, height, width]')
        outs = [] if self.decode_every_timestep else None
        for t in range(X.shape[1]):
            cur_x = X[:, t, :, :, :]

            # iterate through conv and adapt
            for layer in range(len(self.conv_layers)):
                cur_x = self.hks[f'conv_{layer}_{t}'](self.conv_layers[layer](cur_x))

                actvs = actvs_prev.get(layer, self.adapt_layers[layer].get_init_actvs(cur_x, layer))
                cur_x, *new_actvs = self.adapt_layers[layer](cur_x, *actvs)
                cur_x = self.hks[f'adapt_{layer}_{t}'](cur_x)
                new_actvs[0] = self.hks[f'state_{layer}_{t}'](new_actvs[0])
                actvs_prev[layer] = new_actvs

                cur_x = self.relu(cur_x)
                if layer < len(self.conv_layers) - 1:
                    cur_x = self.pool(cur_x)

            # dropout
            cur_x = self.dropout(cur_x)

            # fully connected
            cur_x = cur_x.view(cur_x.size(0), -1)  # [batch, 1024]
            cur_x = self.hook_fc1(self.fc1(cur_x))
            # actvs = actvs_prev.get(4, self.adaptfc1.get_init_actvs(cur_x, 4))
            # cur_x, *new_actvs = self.adaptfc1(cur_x, *actvs)
            # actvs_prev[4] = new_actvs

            if self.decode_every_timestep:
                out = self.decoder(cur_x)
                outs.append(out)
        if self.decode_every_timestep:
            return torch.stack(outs, dim=1)
        # only decode last timestep
        out = self.decoder(cur_x)
        return out

    @staticmethod
    def load_from_checkpoint(path, cfg):
        """
        Load a PyTorch model from a PyTorch Lightning checkpoint.

        Args:
        path (str): Path to the PyTorch Lightning checkpoint.

        Returns:
        torch.nn.HookedRecursiveCNN: The loaded PyTorch model.
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        model_state_dict = checkpoint['state_dict']

        # Instantiate the PyTorch model
        model = HookedRecursiveCNN(**cfg)  # Adjust this according to your model's structure

        # Remove 'model.' prefix from the state dict keys and load into the model
        new_state_dict = {key.replace("model.", ""): value for key, value in model_state_dict.items()}
        model.load_state_dict(new_state_dict)

        return model
