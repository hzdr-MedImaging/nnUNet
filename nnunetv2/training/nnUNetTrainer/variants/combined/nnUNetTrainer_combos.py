import torch
import numpy as np

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
# from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
# from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
# from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainer_noSmooth_500epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainer_noSmooth_250epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainer_noSmooth_125epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 125


