import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSkipWeightLinear(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_skip_epoch = int(self.num_epochs * 0.6)

    def weight_transfer_function(self, x):
        return x

    def calculate_weights(self, target_epoch):
        n_stages = self.configuration_manager.network_arch_init_kwargs['n_stages']
        thresholds = np.r_[-1, np.linspace(0, target_epoch, n_stages)]

        weights = [(self.current_epoch - thresholds[i]) / (thresholds[i + 1] - thresholds[i]) for i in range(n_stages)]
        weights = np.clip(weights, 0, 1)
        weights = np.flip(weights)
        weights = self.weight_transfer_function(weights)
        return weights

    def on_train_epoch_start(self):
        weights = self.calculate_weights(self.target_skip_epoch)
        self.network.set_skip_weights(weights)
        super().on_train_epoch_start()

class nnUNetTrainerSkipWeightSin(nnUNetTrainerSkipWeightLinear):
    def weight_transfer_function(self, x):
        return np.sin(x * np.pi/2)

class nnUNetTrainerSkipWeightStep(nnUNetTrainerSkipWeightLinear):
    def weight_transfer_function(self, x):
        return x >= 1

class nnUNetTrainerSkipWeightConcurrentLinear(nnUNetTrainerSkipWeightLinear):
    def calculate_weights(self, target_epoch):
        n_stages = self.configuration_manager.network_arch_init_kwargs['n_stages']
        thresholds = np.linspace(0, target_epoch, n_stages)

        weights = [(self.current_epoch + 1e-8) / (thresholds[i] + 1e-8) for i in range(n_stages)]
        weights = np.clip(weights, 0, 1)
        weights = np.flip(weights)
        weights = self.weight_transfer_function(weights)
        return weights

class nnUNetTrainerSkipWeightConcurrentHarm(nnUNetTrainerSkipWeightConcurrentLinear):
    def weight_transfer_function(self, x):
        return (1 - np.cos(x * np.pi)) / 2