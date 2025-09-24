import torch

from nnunetv2.training.nnUNetTrainer.variants.benchmarking.nnUNetTrainerBenchmark_5epochs_noDataLoading import (
    nnUNetTrainerBenchmark_5epochs_noDataLoading,
)
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainerDebug(nnUNetTrainerBenchmark_5epochs_noDataLoading):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5
