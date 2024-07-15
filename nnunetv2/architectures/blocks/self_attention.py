from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from nnunetv2.architectures.blocks.spatial_encoding import get_sincos_embeding
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class MHSA(nn.Module):
    """
    After https://link.springer.com/10.1007/s00259-023-06197-1
    """

    def __init__(self, input_channels,
                 conv_op: Type[_ConvNd],
                 num_heads: int = 2,
                 dv: int = None,  # if None, calculated as input_channels / num_heads
                 dk: int = None,  # if None, dk = dv
                 residual: bool = True,
                 position_encoding: bool = True,
                 projection_kernel_size: int = 1,
                 merging_kernel_size: int = 1
                 ):
        super(MHSA, self).__init__()
        self.input_channels = input_channels
        self.num_heads = num_heads
        self.residual = residual
        self.position_encoding = position_encoding
        self.projection_kernel_size = projection_kernel_size
        self.merging_kernel_size = merging_kernel_size
        self.conv_op = conv_op

        if dv is None:
            self.dv = input_channels // num_heads
        else:
            self.dv = dv

        if dk is None:
            self.dk = self.dv
        else:
            self.dk = dk

    def sa_head(self, x, x_enc):
        query = self.conv_op(in_channels=x_enc.shape[1], out_channels=self.dk, kernel_size=self.projection_kernel_size,
                             padding=self.projection_kernel_size // 2, bias=True).forward(x_enc)
        key = self.conv_op(in_channels=x_enc.shape[1], out_channels=self.dk, kernel_size=self.projection_kernel_size,
                           padding=self.projection_kernel_size // 2, bias=True).forward(x_enc)
        value = self.conv_op(in_channels=x.shape[1], out_channels=self.dv, kernel_size=self.projection_kernel_size,
                             padding=self.projection_kernel_size // 2, bias=True).forward(x)

        Q = torch.flatten(query, start_dim=2)  # dim: (b,dk,vox)
        K = torch.flatten(key, start_dim=2)  # dim: (b,dk,vox)
        V = torch.flatten(value, start_dim=2)  # dim: (b,dv,vox)

        Q = nn.functional.normalize(Q, eps=1e-6)
        K = nn.functional.normalize(K, eps=1e-6)

        A = torch.einsum("nqi,nqj->nij", [Q, K]).softmax(dim=-1)
        R = torch.einsum("nij,nvj->nvi", [A, V])  # dim: (b,dv,vox)

        return torch.reshape(R, value.shape)

    def forward(self, x):
        if self.num_heads < 1:
            return x

        if self.position_encoding:
            grid_size = x.shape[2:]  # remove batch and channel dimensions
            pos_matrix = get_sincos_embeding(grid_size=grid_size)
            x_enc = torch.concat([x, pos_matrix], dim=1)
        else:
            x_enc = x

        heads = [self.sa_head(x, x_enc) for i in range(self.num_heads)]

        heads_stacked = torch.concat(heads, dim=1)
        result = self.conv_op(in_channels=heads_stacked.shape[1], out_channels=self.input_channels,
                              kernel_size=self.merging_kernel_size,
                              padding=self.merging_kernel_size // 2, bias=True).forward(heads_stacked)

        if self.residual:
            result = result + x

        return result


class MHSA_interconnect(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 active_stages: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd] = None,
                 num_heads: int = 2,
                 dv: int = None,  # if None, calculated as input_channels / num_heads
                 dk: int = None,  # if None, dk = dv
                 residual: bool = True,
                 position_encoding: bool = True,
                 projection_kernel_size: int = 1,
                 merging_kernel_size: int = 1
                 ):
        super(MHSA_interconnect, self).__init__()
        if isinstance(active_stages, int):
            active_stages = (active_stages,)

        self.features_per_stage = encoder.output_channels
        self.active_stages = active_stages
        self.num_heads = num_heads
        self.residual = residual
        self.position_encoding = position_encoding
        self.projection_kernel_size = projection_kernel_size
        self.merging_kernel_size = merging_kernel_size
        self.conv_op = encoder.conv_op if conv_op is None else conv_op
        self.dv = dv
        self.dk = dk

        self.ops = [nn.Identity() for i in self.features_per_stage]
        for i in self.active_stages:
            self.ops[i] = MHSA(self.features_per_stage[i], self.conv_op, num_heads=self.num_heads, dv=self.dv,
                               dk=self.dk, residual=self.residual, position_encoding=self.position_encoding,
                               projection_kernel_size=self.projection_kernel_size,
                               merging_kernel_size=self.merging_kernel_size)

    def forward(self, skips):
        return [op(skip) for op, skip in zip(self.ops, skips)]


if __name__ == '__main__':
    data = torch.rand((1, 4, 64, 32, 16))

    mhsa = MHSA(4, nn.Conv3d, num_heads=2)

    out = mhsa.forward(data)
    print(out.shape)
