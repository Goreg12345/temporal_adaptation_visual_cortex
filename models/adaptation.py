from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional import accuracy

import pytorch_lightning as pl

from HookedRecursiveCNN import HookedRecursiveCNN
from modules.exponential_decay import ExponentialDecay


class Adaptation(pl.LightningModule):

    def __init__(self, model: HookedRecursiveCNN, lr: float = 1e-3):
        super(Adaptation, self).__init__()

        self.model = model
        self.lr = lr

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        # X [batch, sequence, channel, height, width]
        return self.model(X)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        for i, adapt in enumerate([self.adapt1, self.adapt2, self.adapt3]):
            if not hasattr(adapt, 'params'):
                continue
            metrics = adapt.params()
            for k, v in metrics.items():
                self.log(f'{k}_{i}', v)

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
        loss.backward(*args, **kwargs)
