import sys
from typing import Union, Type, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from nnunetv2.architectures.blocks.spatial_encoding import get_sincos_embeding
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class SAHead(nn.Module):
    def __init__(self, input_channels: int,
                 input_channels_with_enc: int,
                 conv_op: Type[_ConvNd],
                 dv: int,  # if None, calculated as input_channels / num_heads
                 dk: int,  # if None, dk = dv
                 projection_kernel_size: int = 1,
                 save_attention: bool = True,
                 qk_norm_type: str = "l2"
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.input_channels_with_enc = input_channels_with_enc
        self.projection_kernel_size = projection_kernel_size
        self.save_attention = save_attention
        self.attention_matrix = None
        self.qk_norm_type = qk_norm_type
        self.conv_op = conv_op
        self.dv = dv
        self.dk = dk
        if qk_norm_type == "l2scaled":
            self.softmax_scale = torch.nn.Parameter(torch.ones(1))

        self.proj_q = self.conv_op(in_channels=self.input_channels_with_enc,
                                   out_channels=self.dk, kernel_size=self.projection_kernel_size,
                                   padding=self.projection_kernel_size // 2, bias=True)
        self.proj_k = self.conv_op(in_channels=self.input_channels_with_enc,
                                   out_channels=self.dk, kernel_size=self.projection_kernel_size,
                                   padding=self.projection_kernel_size // 2, bias=True)
        self.proj_v = self.conv_op(in_channels=self.input_channels, out_channels=self.dv,
                                   kernel_size=self.projection_kernel_size,
                                   padding=self.projection_kernel_size // 2, bias=True)

    def forward(self, x, x_enc):
        query = self.proj_q(x_enc)
        key = self.proj_k(x_enc)
        value = self.proj_v(x)

        Q = torch.flatten(query, start_dim=2)  # dim: (b,dk,vox)
        K = torch.flatten(key, start_dim=2)  # dim: (b,dk,vox)
        V = torch.flatten(value, start_dim=2)  # dim: (b,dv,vox)

        if self.qk_norm_type == "l2":
            Q = nn.functional.normalize(Q, eps=1e-6)
            K = nn.functional.normalize(K, eps=1e-6)
        elif self.qk_norm_type == "l2scaled":
            Q = nn.functional.normalize(Q, eps=1e-6)
            K = nn.functional.normalize(K, eps=1e-6)
            Q = Q * self.softmax_scale
        elif self.qk_norm_type == "l2x2":
            Q = nn.functional.normalize(Q, eps=1e-6)
            K = nn.functional.normalize(K, eps=1e-6)
            Q = Q * (2 ** (1 / 2))
            K = K * (2 ** (1 / 2))
        elif self.qk_norm_type == "l2x3":
            Q = nn.functional.normalize(Q, eps=1e-6)
            K = nn.functional.normalize(K, eps=1e-6)
            Q = Q * (3 ** (1 / 2))
            K = K * (3 ** (1 / 2))
        elif self.qk_norm_type == "l2x4":
            Q = nn.functional.normalize(Q, eps=1e-6)
            K = nn.functional.normalize(K, eps=1e-6)
            Q = Q * 2
            K = K * 2
        elif self.qk_norm_type == "dk":
            Q = Q / (self.dk ** (1 / 4))
            K = K / (self.dk ** (1 / 4))
        else:
            sys.exit("Unsupported QK-normalization")

        A = torch.einsum("nqi,nqj->nij", [Q, K]).softmax(dim=-1) # softmax along j: keys
        R = torch.einsum("nij,nvj->nvi", [A, V])  # dim: (b,dv,vox)

        if self.save_attention:
            self.attention_matrix = A.detach()
            batch_dim = value.shape[:1]
            spatial_dims = value.shape[2:]
            self.attention_matrix = self.attention_matrix.reshape(batch_dim + spatial_dims + spatial_dims)

        return torch.reshape(R, value.shape)

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
                 merging_kernel_size: int = 1,
                 merging_bias: bool = True,
                 position_encoding_dim: int = 96,
                 save_attention: bool = True,
                 qk_norm_type: str = "l2"
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.num_heads = num_heads
        self.residual = residual
        self.position_encoding = position_encoding
        self.projection_kernel_size = projection_kernel_size
        self.merging_kernel_size = merging_kernel_size
        self.position_encoding_dim = position_encoding_dim
        self.save_attention = save_attention
        self.conv_op = conv_op
        self.qk_norm_type = qk_norm_type

        if dv is None:
            self.dv = input_channels // num_heads
        else:
            self.dv = dv

        if dk is None:
            self.dk = self.dv
        else:
            self.dk = dk

        self.proj_merge = self.conv_op(in_channels=self.dv * num_heads, out_channels=self.input_channels,
                                       kernel_size=self.merging_kernel_size,
                                       padding=self.merging_kernel_size // 2, bias=merging_bias)
        self.sa_heads = [SAHead(self.input_channels,
                                self.input_channels + self.position_encoding_dim * self.position_encoding,
                                self.conv_op, self.dv, self.dk, self.projection_kernel_size, self.save_attention,
                                self.qk_norm_type)
                         for i in range(self.num_heads)]
        self.sa_heads = nn.ModuleList(self.sa_heads)

    def forward(self, x):
        if self.num_heads < 1:
            return x

        if self.position_encoding:
            grid_size = x.shape[2:]  # remove batch and channel dimensions
            pos_matrix = get_sincos_embeding(grid_size=grid_size, batch_size=x.shape[0],
                                             embed_dim=self.position_encoding_dim, device=x.device)
            x_enc = torch.concat([x, pos_matrix], dim=1)
        else:
            x_enc = x

        head_outputs = [self.sa_heads[i](x, x_enc) for i in range(self.num_heads)]

        outputs_stacked = torch.concat(head_outputs, dim=1)
        result = self.proj_merge(outputs_stacked)

        if self.residual:
            result = result + x

        return result

    def get_attention_list(self):
        return [self.sa_heads[i].attention_matrix for i in range(self.num_heads)]

    def compute_memory(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """

        # let's sum all the operations
        output = np.int64(0)
        # pos_matrix
        output += np.prod([self.position_encoding_dim, *input_size], dtype=np.int64)
        # x_enc
        output += np.prod([self.input_channels + self.position_encoding_dim, *input_size], dtype=np.int64)

        return output


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
                 merging_kernel_size: int = 1,
                 merging_bias: bool = True,
                 qk_norm_type: str = "l2"
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
        self.qk_norm_type = qk_norm_type
        self.conv_op = encoder.conv_op if conv_op is None else conv_op
        self.dv = dv
        self.dk = dk

        self.ops = [nn.Identity() for i in self.features_per_stage]
        for i in self.active_stages:
            self.ops[i] = MHSA(self.features_per_stage[i], self.conv_op, num_heads=self.num_heads, dv=self.dv,
                               dk=self.dk, residual=self.residual, position_encoding=self.position_encoding,
                               projection_kernel_size=self.projection_kernel_size,
                               merging_kernel_size=self.merging_kernel_size, merging_bias=merging_bias,
                               qk_norm_type=qk_norm_type)
        self.ops = nn.ModuleList(self.ops)

    def forward(self, skips):
        return [op(skip) for op, skip in zip(self.ops, skips)]

    def get_all_attention_maps(self):
        return [self.ops[i].get_attention_list() for i in self.active_stages]

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


if __name__ == '__main__':
    data = torch.rand((3, 4, 64, 32, 16))

    mhsa = MHSA(4, nn.Conv3d, num_heads=2, qk_norm_type="l2scaled")
    print(mhsa)
    [print(name) for name, _ in mhsa.named_children()]
    print(mhsa(data).shape)
    [print(A.shape) for A in mhsa.get_attention_list()]

    # out = mhsa.forward(data)
    # print(out.shape)
    print()
    encoder = ResidualEncoder(2, 6, (32, 64, 128, 256, 320, 320), nn.Conv3d, 3,
                              ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                              (1, 3, 4, 6, 6, 6), True, nn.modules.instancenorm.InstanceNorm3d,
                              nonlin=nn.ReLU,
                              return_skips=True, disable_default_stem=False, stem_channels=None)
    print(encoder)
    mhsa_ic = MHSA_interconnect(encoder, (-1, -2))
    print(mhsa_ic)
    [print(name) for name, _ in mhsa_ic.named_children()]
