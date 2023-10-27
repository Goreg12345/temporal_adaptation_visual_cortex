from typing import Any

import torch
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional import accuracy

import pytorch_lightning as pl

from HookedRecursiveCNN import HookedRecursiveCNN
from metrics.sparsity import Sparsity
from metrics.update_to_weight_norm import RelativeGradientUpdateNorm
from metrics.dead_neurons_counter import DeadNeuronMetric


class Adaptation(pl.LightningModule):

    def __init__(self, model: HookedRecursiveCNN, lr: float = 1e-3):
        super(Adaptation, self).__init__()

        self.model = model
        self.lr = lr

        self.loss = nn.CrossEntropyLoss()

        # METRICS
        self.update_to_weight_metric = RelativeGradientUpdateNorm(self.model.named_parameters())
        self.dead_neurons_counter = DeadNeuronMetric(n_adapt_layers=len(model.adapt_layers), n_timesteps=model.t_steps)
        self.dead_feature_maps_counter = DeadNeuronMetric(n_adapt_layers=len(model.adapt_layers), n_timesteps=model.t_steps, per='map')
        self.sparsity = Sparsity(n_adapt_layers=len(model.adapt_layers), timestep=-1)

    def forward(self, X):
        # X [batch, sequence, channel, height, width]
        return self.model(X)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, cache = self.model.run_with_cache(x)
        loss = self.loss(y_hat, y)

        # METRICS
        self.log('train_loss', loss)
        self.dead_neurons_counter.update(cache)
        self.dead_feature_maps_counter.update(cache)
        self.sparsity.update(cache)
        self.log_dict(self.sparsity.compute())

        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        for i, adapt in enumerate(self.model.adapt_layers):
            if not hasattr(adapt, 'params'):
                continue
            metrics = adapt.params()
            for k, v in metrics.items():
                # if v is scalar
                if v.dim() == 0:
                    self.log(f'{k}_{i}', v)
                # if v is tensor
                else:
                    # log histogram
                    if isinstance(self.logger, TensorBoardLogger):
                        self.logger.add_histogram(f'{k}_{i}', v, self.global_step)

                    else:
                        self.logger.experiment.log({f'{k}_{i}': wandb.Histogram(v.detach().cpu())})

        self.update_to_weight_metric.update(self.model.named_parameters())
        avg_update_to_weight_norm = self.update_to_weight_metric.compute()
        for k, v in avg_update_to_weight_norm.items():
            self.log(f'avg_update_to_weight_norm/{k}', v)

    def on_train_epoch_end(self) -> None:
        # log dead neurons
        self.log_dict(self.dead_neurons_counter.compute())
        self.log_dict(self.dead_feature_maps_counter.compute())

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
        return
