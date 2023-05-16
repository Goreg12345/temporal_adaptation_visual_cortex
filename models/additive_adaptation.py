from typing import Any

import torch
import os
import torch.nn as nn
from torchmetrics.functional import accuracy

from models.module_exp_decay import module_exp_decay
import pytorch_lightning as pl


class AdditiveAdaptation(pl.LightningModule):

    def __init__(self, t_steps):
        super(AdditiveAdaptation, self).__init__()

        # training variables
        self.t_steps = t_steps

        # activation functions, pooling and dropout layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

        # placeholders
        init = torch.zeros(4)

        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.sconv1 = module_exp_decay(32, 24, 24, init[0], True, init[0], True)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.sconv2 = module_exp_decay(32, 8, 8, init[1], True, init[1], True)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.sconv3 = module_exp_decay(32, 2, 2, init[2], True, init[2], True)

        # fc 1
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.sfc1 = module_exp_decay('none', 'none', 1024, init[3], True, init[3], True)

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024,
                                 out_features=10)  # only saves the output from the last timestep to train
        self.lr = 0.001

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):

        """ Feedforward sweep.

        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer

        """

        # X [batch, sequence, channel, height, width]

        actvs_prev = {}
        state_prev = {}

        for i in range(X.shape[1]):
            cur_x = X[:, i, :, :, :]

            cur_x = self.conv1(cur_x)
            actvs = actvs_prev.get(1,torch.zeros_like(cur_x))
            state = state_prev.get(1,torch.zeros_like(cur_x))

            s_updt, s_beta_updt = self.sconv1(actvs, state)
            state_prev[1] = s_updt
            actvs_prev[1] = self.relu(torch.subtract(cur_x, s_beta_updt))
            cur_x = self.pool(actvs_prev[1])

            # conv2
            cur_x = self.conv2(cur_x)
            actvs = actvs_prev.get(2,torch.zeros_like(cur_x))
            state = state_prev.get(2,torch.zeros_like(cur_x))

            s_updt, s_beta_updt = self.sconv2(actvs, state)
            state_prev[2] = s_updt
            actvs_prev[2] = self.relu(torch.subtract(cur_x, s_beta_updt))
            cur_x = self.pool(actvs_prev[2])

            # conv3
            cur_x = self.conv3(cur_x)
            actvs = actvs_prev.get(3,torch.zeros_like(cur_x))
            state = state_prev.get(3,torch.zeros_like(cur_x))

            s_updt, s_beta_updt = self.sconv3(actvs, state)
            state_prev[3] = s_updt
            actvs_prev[3] = self.relu(torch.subtract(cur_x, s_beta_updt))
            cur_x = actvs_prev[3]

            # dropout
            cur_x = self.dropout(cur_x)

            # fully connected
            cur_x = cur_x.view(cur_x.size(0), -1)  # [batch, 1024]
            cur_x = self.fc1(cur_x)
            actvs = actvs_prev.get(4,torch.zeros_like(cur_x))
            state = state_prev.get(4,torch.zeros_like(cur_x))

            s_updt, s_beta_updt = self.sfc1(actvs, state)
            state_prev[4] = s_updt
            actvs_prev[4] = torch.subtract(cur_x, s_beta_updt)

        # only decode last timestep
        out = self.decoder(actvs_prev[4])
        # actvs[4] = torch.cat((actvs[4],self.decoder(actvs[3][t+1])))

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
