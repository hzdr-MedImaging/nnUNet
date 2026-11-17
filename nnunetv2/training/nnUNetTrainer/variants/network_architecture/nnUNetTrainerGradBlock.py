import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerGradBlockSkips(nnUNetTrainer):
    def on_train_epoch_start(self):
        n_stages = self.configuration_manager.network_arch_init_kwargs['n_stages']
        blocks = [True] * n_stages
        blocks[-1] = False
        self.network.set_grad_blocks(blocks)
        super().on_train_epoch_start()

class nnUNetTrainerGradBlockSkipsSteps(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_unblock_epoch = int(self.num_epochs * 0.15)

    def on_train_epoch_start(self):
        n_stages = self.configuration_manager.network_arch_init_kwargs['n_stages']
        thresholds = np.linspace(self.target_unblock_epoch,0, n_stages).tolist()
        blocks = [self.current_epoch < thr for thr in thresholds]
        self.network.set_grad_blocks(blocks)
        super().on_train_epoch_start()