from typing import Tuple, Union, Type, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder


class AuxFCHead(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 active_stages: Union[int, List[int], Tuple[int, ...]],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 pool: Union[Type[_AdaptiveMaxPoolNd], Type[_AdaptiveAvgPoolNd]] = None,
                 ):
        super(AuxFCHead, self).__init__()
        if isinstance(active_stages, int):
            active_stages = (active_stages,)

        self.features_per_stage = encoder.output_channels
        self.active_stages = active_stages
        self.conv_op = encoder.conv_op if conv_op is None else conv_op

        #for nnd
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        self.ops = [nn.Identity() for i in self.features_per_stage]
        for i in self.active_stages:
            self.ops[i] = MHSA(self.features_per_stage[i], self.conv_op, num_heads=self.num_heads, dv=self.dv,
                               dk=self.dk, residual=self.residual, position_encoding=self.position_encoding,
                               projection_kernel_size=self.projection_kernel_size,
                               merging_kernel_size=self.merging_kernel_size, merging_bias=merging_bias,
                               qk_norm_type=qk_norm_type, save_attention=save_attention,
                               nnd=nnd, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                               dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=
                               nonlin_kwargs)
        self.ops = nn.ModuleList(self.ops)

    def forward(self, skips):
        return [op(skip) for op, skip in zip(self.ops, skips)]


    def compute_memory(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes.
        skip_sizes = []
        for s in range(len(encoder.strides)):
            skip_sizes.append([i // j for i, j in zip(input_size, encoder.strides[s])])
            input_size = skip_sizes[-1]
        print(skip_sizes)

        assert len(skip_sizes) == len(self.features_per_stage)

        # go over active stages and sum up the memory
        output = np.int64(0)
        for s in self.active_stages:
            output += self.ops[s].compute_memory(skip_sizes[s])
        return output