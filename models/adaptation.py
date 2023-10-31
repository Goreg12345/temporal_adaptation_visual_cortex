from typing import Any

import torch
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional import accuracy

import pytorch_lightning as pl

from HookedRecursiveCNN import HookedRecursiveCNN
from metrics.accuracy_per_setting import AccuracyPerSetting
from metrics.accuracy_difference import AccuracyDifference
from metrics.cumulative_activation_diff import CumulativeActivationDiff
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
        self.acc_per_setting = AccuracyPerSetting(contrasts=[0.2, 0.4, 0.6, 0.8, 1.0], repeated_noise=[True, False])
        self.acc_diff = AccuracyDifference(contrasts=[0.2, 0.4, 0.6, 0.8, 1.0], repeated_noise=[True, False])
        self.cum_actv_diff = CumulativeActivationDiff(contrasts=[0.2, 0.4, 0.6, 0.8, 1.0], repeated_noise=[True, False], n_layers=len(model.adapt_layers))

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

        if batch_idx % 100 == 0:  # because this is timeconsuming and not super important
            # activations histogram
            for layer in range(len(self.model.adapt_layers)):
                self.logger.experiment.log({f'actvs/conv_{layer}_0': wandb.Histogram(cache[f'hks.conv_{layer}_0'].detach().cpu())})
                last_timestep = max([int(key.split('_')[-1]) for key in cache.keys() if 'adapt' in key])
                self.logger.experiment.log({f'actvs/conv_{layer}_last': wandb.Histogram(cache[f'hks.conv_{layer}_{last_timestep}'].detach().cpu())})
                self.logger.experiment.log({f'actvs/adapt_{layer}_0': wandb.Histogram(cache[f'hks.adapt_{layer}_0'].detach().cpu())})
                self.logger.experiment.log({f'actvs/adapt_{layer}_last': wandb.Histogram(cache[f'hks.adapt_{layer}_{last_timestep}'].detach().cpu())})

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
        x, y, contrast, rep_noise = batch
        y_hat, cache = self.model.run_with_cache(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        # accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)
        self.log('val_acc', acc)

        self.acc_per_setting.update(y_hat, y, contrast, rep_noise)
        self.acc_diff.update(y_hat, y, contrast, rep_noise)
        self.cum_actv_diff.update(cache, contrast, rep_noise)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.acc_per_setting.compute())
        self.log_dict(self.acc_diff.compute())
        self.log_dict(self.cum_actv_diff.compute())

    def test_step(self, batch, batch_idx):
        x, y, contrast, rep_noise = batch
        y_hat, cache = self.model.run_with_cache(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        # accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)
        self.log('test_acc', acc)

        self.acc_per_setting.update(y_hat, y, contrast, rep_noise)
        self.acc_diff.update(y_hat, y, contrast, rep_noise)
        self.cum_actv_diff.update(cache, contrast, rep_noise)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.acc_per_setting.compute())
        self.log_dict(self.acc_diff.compute())
        self.log_dict(self.cum_actv_diff.compute())

    def backward(self, loss, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)
        return
