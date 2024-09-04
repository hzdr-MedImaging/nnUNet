import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.training.nnUNetTrainer.variants.sampling.nnUNetTrainer_probabilisticOversampling import nnUNetTrainer_probabilisticOversampling_010
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.lr_scheduler.weibull_lr import WeibullLRScheduler


class nnUNetTrainer_noSmooth_pretrained(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainer_noSmooth_finetune_e250_lr3(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        self.initial_lr = 1e-3


class nnUNetTrainer_noSmooth_weibull(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.max_lr = self.initial_lr

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = WeibullLRScheduler(optimizer, max_steps=self.num_epochs, max_lr=self.max_lr)
        return optimizer, lr_scheduler


class nnUNetTrainer_noSmooth_finetune_weibull_e100_lr5e3(nnUNetTrainer_noSmooth_weibull):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.max_lr = 5e-3


class nnUNetTrainer_noSmooth_finetune_weibull_e200_lr1e3(nnUNetTrainer_noSmooth_weibull):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200
        self.max_lr = 1e-3


class nnUNetTrainer_noSmooth_probOS01_finetune_weibull_e200_lr1e3(nnUNetTrainer_probabilisticOversampling_010,
                                                                  nnUNetTrainer_noSmooth_finetune_weibull_e200_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainer_noSmooth_finetune_weibull_e500_lr1e3(nnUNetTrainer_noSmooth_weibull):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.max_lr = 1e-3


class nnUNetTrainer_noSmooth_finetune_weibull_e200_lr2e3(nnUNetTrainer_noSmooth_weibull):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200
        self.max_lr = 2e-3






