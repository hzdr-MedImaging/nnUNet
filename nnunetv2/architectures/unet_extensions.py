from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.architectures.blocks.self_attention import MHSA_interconnect


class ResEncUNetWithSA(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 num_sa_heads: int = 2,
                 sa_stage_indices: Union[int, List[int], Tuple[int, ...]] = -1,
                 residual_sa: bool = True,
                 sa_merging_bias: bool = True,
                 qk_norm_type: str = "l2",
                 sa_nnd: bool = False,
                 save_attention: bool = False
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.interconnect = MHSA_interconnect(self.encoder, active_stages=sa_stage_indices, num_heads=num_sa_heads,
                                              residual=residual_sa, merging_bias=sa_merging_bias,
                                              qk_norm_type=qk_norm_type, nnd=sa_nnd, save_attention=save_attention)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        print("Active SA stages:", sa_stage_indices)
        print("Num SA heads:", num_sa_heads)

    def forward(self, x):
        skips = self.encoder(x)
        skips = self.interconnect(skips)
        #[print(A.shape) for att_list in self.interconnect.get_all_attention_mats() for A in att_list]
        return self.decoder(skips)

    def get_all_attention_maps(self):
        return self.interconnect.get_all_attention_maps()

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = ResEncUNetWithSA(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2),
                             3,
                             (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True,
                             sa_stage_indices=(-1, -2))

    out = model(data)[0]

    print(out.shape, "->", model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    model = ResEncUNetWithSA(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2),
                             (2, 2, 2, 2, 2, 2, 2, 2), 3,
                             (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU,
                             deep_supervision=True)

    out = model(data)[0]

    print(out.shape, "->", model.compute_conv_feature_map_size(data.shape[2:]))
