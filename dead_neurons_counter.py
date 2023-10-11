import torch
from torchmetrics import Metric


class DeadNeuronMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("dead_neurons", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("dead_gradients", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_neurons", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, activations, gradients):
        # Update counts
        self.dead_neurons += torch.sum(activations == 0)
        self.dead_gradients += torch.sum(gradients == 0)
        self.total_neurons += torch.numel(activations)

    def compute(self):
        # Compute percentages
        dead_neuron_percentage = self.dead_neurons.float() / self.total_neurons
        dead_gradient_percentage = self.dead_gradients.float() / self.total_neurons
        return dead_neuron_percentage, dead_gradient_percentage
