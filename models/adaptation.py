from typing import Any

import torch
import torch.nn as nn
from torchmetrics.functional import accuracy

import pytorch_lightning as pl

from modules.exponential_decay import ExponentialDecay


class Adaptation(pl.LightningModule):

    def __init__(self, t_steps, layer_kwargs, adaptation_kwargs, lr, adaptation_module=ExponentialDecay,):
        super(Adaptation, self).__init__()

        # training variables
        self.t_steps = t_steps

        # activation functions, pooling and dropout layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

        # conv1
        self.conv1 = nn.Conv2d(**layer_kwargs[0])
        self.adapt1 = adaptation_module(**adaptation_kwargs[0])

        # conv2
        self.conv2 = nn.Conv2d(**layer_kwargs[1])
        self.adapt2 = adaptation_module(**adaptation_kwargs[1])

        # conv3
        self.conv3 = nn.Conv2d(**layer_kwargs[2])
        self.adapt3 = adaptation_module(**adaptation_kwargs[2])

        # fc 1
        self.fc1 = nn.Linear(**layer_kwargs[3])
        self.adaptfc1 = adaptation_module(**adaptation_kwargs[3])

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024,
                                 out_features=10)  # only saves the output from the last timestep to train
        self.lr = lr

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):

        """ Feedforward sweep.

        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer

        """

        # X [batch, sequence, channel, height, width]

        actvs_prev = {}

        for i in range(X.shape[1]):
            cur_x = X[:, i, :, :, :]

            cur_x = self.conv1(cur_x)
            actvs = actvs_prev.get(1, self.adapt1.get_init_actvs(cur_x, 1))
            cur_x, *new_actvs = self.adapt1(cur_x, *actvs)
            actvs_prev[1] = new_actvs
            cur_x = self.pool(cur_x)

            # conv2
            cur_x = self.conv2(cur_x)
            actvs = actvs_prev.get(2,self.adapt2.get_init_actvs(cur_x, 2))
            cur_x, *new_actvs = self.adapt2(cur_x, *actvs)
            actvs_prev[2] = new_actvs
            cur_x = self.pool(cur_x)

            # conv3
            cur_x = self.conv3(cur_x)
            actvs = actvs_prev.get(3,self.adapt3.get_init_actvs(cur_x, 3))
            cur_x, *new_actvs = self.adapt3(cur_x, *actvs)
            actvs_prev[3] = new_actvs

            # dropout
            cur_x = self.dropout(cur_x)

            # fully connected
            cur_x = cur_x.view(cur_x.size(0), -1)  # [batch, 1024]
            cur_x = self.fc1(cur_x)
            actvs = actvs_prev.get(4, self.adaptfc1.get_init_actvs(cur_x, 4))
            cur_x, *new_actvs = self.adaptfc1(cur_x, *actvs)
            actvs_prev[4] = new_actvs

        # only decode last timestep
        out = self.decoder(cur_x)
        return out

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        # accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        # accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)
        self.log('test_acc', acc)
        return loss

    def backward(self, loss, *args: Any, **kwargs: Any) -> None:
        loss.backward()
