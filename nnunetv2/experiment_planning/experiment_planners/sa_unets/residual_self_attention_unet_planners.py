import warnings

import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from nnunetv2.architectures.unet_extensions import ResEncUNetWithSA
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import ResEncUNetPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props


class ResEncUNetWithSAPlanner(ResEncUNetPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetWithSAPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResEncUNetWithSA

        sa_3d_configs = {
            "3d_fullres_sa4": {
                "inherits_from": "3d_fullres",
                "architecture": {
                    "arch_kwargs": {
                        "num_sa_heads": 4
                    }
                }
            }
        }
        self.extra_3d_fullres_configs.update(sa_3d_configs)

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset,
                                                  _cache)

        plan['architecture']['arch_kwargs']['sa_stage_indices'] = [-1, -2, -3]
        plan['architecture']['arch_kwargs']['num_sa_heads'] = 2
        plan['architecture']['arch_kwargs']['sa_merging_bias'] = False

        return plan


class nnUNetPlannerResEncSAM(ResEncUNetWithSAPlanner):
    """
    Target is ~9-11 GB VRAM max -> older Titan, RTX 2080ti
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncSAUNetMPlans',
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


class nnUNetPlannerResEncSAL(ResEncUNetWithSAPlanner):
    """
    Target is ~24 GB VRAM max -> RTX 4090, Titan RTX, Quadro 6000
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncSAUNetLPlans',
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


class nnUNetPlannerResEncSAXL(ResEncUNetWithSAPlanner):
    """
    Target is 40 GB VRAM max -> A100 40GB, RTX 6000 Ada Generation
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncSAUNetXLPlans',
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


if __name__ == '__main__':
    # we know both of these networks run with batch size 2 and 12 on ~8-10GB, respectively
    net = ResidualEncoderUNet(input_channels=1, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
                              conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((128, 128, 128)))  # -> 558319104. The value you see above was finetuned
    # from this one to match the regular nnunetplans more closely

    net = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((512, 512)))  # -> 129793792
