import torch
import numpy as np

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerGradBlock import \
    nnUNetTrainerGradBlockSkips, nnUNetTrainerGradBlockSkipsSteps
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerSkipWeights import \
    nnUNetTrainerSkipWeightLinear, nnUNetTrainerSkipWeightSin, nnUNetTrainerSkipWeightStep, \
    nnUNetTrainerSkipWeightConcurrentLinear, nnUNetTrainerSkipWeightConcurrentHarm
from nnunetv2.training.nnUNetTrainer.variants.sampling.nnUNetTrainer_probabilisticOversampling import nnUNetTrainer_probabilisticOversampling
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerDeepSupervisionVariants import nnUNetTrainer_betaBinomDS, nnUNetTrainer_betaBinomDSm1, nnUNetTrainer_betaBinomDSm2
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision


# alias for nnUNetTrainerDiceCELoss_noSmooth
class nnUNetTrainer_noSmooth(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_2000epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000


class nnUNetTrainer_noSmooth_500epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainer_noSmooth_250epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainer_noSmooth_125epoch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 125


class nnUNetTrainer_noSmooth_betaBinomDS(nnUNetTrainerDiceCELoss_noSmooth, nnUNetTrainer_betaBinomDS):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_betaBinomDSm1(nnUNetTrainerDiceCELoss_noSmooth, nnUNetTrainer_betaBinomDSm1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_betaBinomDSm2(nnUNetTrainerDiceCELoss_noSmooth, nnUNetTrainer_betaBinomDSm2):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_noDS(nnUNetTrainerDiceCELoss_noSmooth, nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_500epoch_probOS10(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.1


class nnUNetTrainer_noSmooth_500epoch_probOS33(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.33


class nnUNetTrainer_noSmooth_probOS10(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.1


class nnUNetTrainer_noSmooth_probOS33(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.33


class nnUNetTrainer_noSmooth_probOS67(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.67


class nnUNetTrainer_noSmooth_betaBinomDS_probOS33(nnUNetTrainer_noSmooth_probOS33, nnUNetTrainer_betaBinomDS):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_betaBinomDSm1_probOS33(nnUNetTrainer_noSmooth_probOS33, nnUNetTrainer_betaBinomDSm1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_betaBinomDSm2_probOS33(nnUNetTrainer_noSmooth_probOS33, nnUNetTrainer_betaBinomDSm2):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_noSmooth_noDS_probOS33(nnUNetTrainer_noSmooth_probOS33, nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

class nnUNetTrainer_nS_SkipWeightLinear(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightLinear):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

class nnUNetTrainer_nS_SkipWeightSin(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightSin):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

class nnUNetTrainer_nS_SkipWeightStep(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightStep):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

class nnUNetTrainer_nS_SkipWeightStep150(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightStep):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_skip_epoch = 150

class nnUNetTrainer_nS_SkipWeightStep50(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightStep):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_skip_epoch = 50

class nnUNetTrainer_nS_SkipWeightConcurrentLinear50(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightConcurrentLinear):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_skip_epoch = 50

class nnUNetTrainer_nS_SkipWeightConcurrentHarm50(nnUNetTrainer_noSmooth, nnUNetTrainerSkipWeightConcurrentHarm):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_skip_epoch = 50

class nnUNetTrainer_nS_GradBlockSkips(nnUNetTrainer_noSmooth, nnUNetTrainerGradBlockSkips):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

class nnUNetTrainer_nS_GradBlockSkipsSteps(nnUNetTrainer_noSmooth, nnUNetTrainerGradBlockSkipsSteps):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
