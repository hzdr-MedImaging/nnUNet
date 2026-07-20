import warnings

import numpy as np
from typing import Union, List, Tuple

from nnunetv2.architectures.operations.global_pooling import GlobalAvgPool3d, GlobalAvgPool2d, GlobalLpPoolTrainable3d, \
    GlobalLpPoolTrainable2d
from nnunetv2.architectures.unet_extensions import ResEncUNetAuxTask, ResEncUNetAuxEnh
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import ResEncUNetPlanner

# Does not really respect the memory limits. But aux heads should be lightweight anyway
class ResEncUNetAuxTaskPlanner(ResEncUNetPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncAuxTaskUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResEncUNetAuxTask

        aux_3d_configs = {
            "3d_fullres_nograd": {
                "inherits_from": "3d_fullres",
                "architecture": {
                    "arch_kwargs": {
                        "block_grad": True
                    }
                }
            },
            "3d_fullres_h6464": {
                "inherits_from": "3d_fullres",
                "architecture": {
                    "arch_kwargs": {
                        "aux_hidden_features": [64,64]
                    }
                }
            },
            "3d_fullres_as2": {
                "inherits_from": "3d_fullres",
                "architecture": {
                    "arch_kwargs": {
                        "aux_active_stages": [-1,-2]
                    }
                }
            },
            "3d_fullres_fgo": {
                "inherits_from": "3d_fullres",
                "aux_fg_only": True,
            }
        }
        self.extra_3d_fullres_configs.update(aux_3d_configs)

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset,
                                                  _cache)

        is_3d = len(plan['patch_size']) == 3

        pool = GlobalLpPoolTrainable3d if is_3d else GlobalLpPoolTrainable2d
        norm = nn.LayerNorm
        dropout = None
        nonlin = nn.LeakyReLU

        plan['architecture']['arch_kwargs'].update({
            'aux_active_stages': [-1, -2, -3],
            'aux_hidden_features':  64,
            'aux_output_logits':  1,
            'aux_pool_op':  pool.__module__ + '.' + pool.__name__,
            'aux_pool_op_kwargs': {},
            'aux_norm_op':  norm.__module__ + '.' + norm.__name__,
            'aux_norm_op_kwargs': {'eps': 1e-5},
            'aux_dropout_op': None if dropout is None else dropout.__module__ + '.' + dropout.__name__,
            'aux_dropout_op_kwargs': {},
            'aux_nonlin': nonlin.__module__ + '.' + nonlin.__name__,
            'aux_nonlin_kwargs': {'inplace': True},
            'aux_final_act': None,
            'aux_final_act_kwargs': {},
            'aux_block_grad': False,
        })
        plan['architecture']['_kw_requires_import'] += ('aux_pool_op', 'aux_norm_op', 'aux_dropout_op',
                                                        'aux_nonlin', 'aux_final_act')
        plan['aux_loss_weight'] = 1
        plan['aux_fg_only'] = False

        return plan


class nnUNetPlannerResEncAuxM(ResEncUNetAuxTaskPlanner):
    """
    Target is ~9-11 GB VRAM max -> older Titan, RTX 2080ti
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncAuxTaskUNetMPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 8:
            warnings.warn("WARNING: You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 8, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 8

        # this is supposed to give the same GPU memory requirement as the default nnU-Net
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.max_dataset_covered = 1


class nnUNetPlannerResEncAuxL(ResEncUNetAuxTaskPlanner):
    """
    Target is ~24 GB VRAM max -> RTX 4090, Titan RTX, Quadro 6000
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncAuxTaskUNetLPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 24:
            warnings.warn("WARNING: You are running nnUNetPlannerL with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 24, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 24

        self.UNet_reference_val_3d = 2100000000  # 1840000000
        self.UNet_reference_val_2d = 380000000  # 352666667
        self.max_dataset_covered = 1


class nnUNetPlannerResEncAuxXL(ResEncUNetAuxTaskPlanner):
    """
    Target is 40 GB VRAM max -> A100 40GB, RTX 6000 Ada Generation
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncAuxTaskUNetXLPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 40:
            warnings.warn("WARNING: You are running nnUNetPlannerXL with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 40, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 40

        self.UNet_reference_val_3d = 3600000000
        self.UNet_reference_val_2d = 560000000
        self.max_dataset_covered = 1


class ResEncUNetAuxEnhPlanner(ResEncUNetAuxTaskPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncAuxTaskUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResEncUNetAuxEnh

class nnUNetPlannerResEncAuxEnhM(ResEncUNetAuxEnhPlanner):
    """
    Target is ~9-11 GB VRAM max -> older Titan, RTX 2080ti
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncAuxEnhUNetMPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 8:
            warnings.warn("WARNING: You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 8, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 8

        # this is supposed to give the same GPU memory requirement as the default nnU-Net
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.max_dataset_covered = 1