import sys
from typing import Union, Type, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.architectures.blocks.spatial_encoding import get_sincos_embeding
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class DropoutNormNonlin(nn.Module):
    def __init__(self,
                 output_channels: int,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(DropoutNormNonlin, self).__init__()

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)


class QKNormalize(nn.Module):
    def __init__(self,
                 qk_norm_type: str,
                 dk: int
                 ):
        super(QKNormalize, self).__init__()

        if qk_norm_type == "l2scaled":
            self.softmax_scale = torch.nn.Parameter(torch.ones(1))
        if qk_norm_type == "l2scaled1":
            self.softmax_scale = torch.nn.Parameter(torch.ones(1))
        if qk_norm_type == "l2scaled2":
            self.softmax_scale = torch.nn.Parameter(torch.zeros(1))


        def norm_func(Q):
            if qk_norm_type == "l2":
                Q = nn.functional.normalize(Q, eps=1e-6)
            elif qk_norm_type == "l2scaled":
                Q = nn.functional.normalize(Q, eps=1e-6)
                Q = Q * (self.softmax_scale ** (1 / 2))
            elif qk_norm_type == "l2scaled1" or qk_norm_type == "l2scaled2":
                Q = nn.functional.normalize(Q, eps=1e-6)
                Q = Q * (self.softmax_scale ** 2 + 1) ** (1 / 2)
            elif qk_norm_type == "l2x2":
                Q = nn.functional.normalize(Q, eps=1e-6)
                Q = Q * (2 ** (1 / 2))
            elif qk_norm_type == "l2x3":
                Q = nn.functional.normalize(Q, eps=1e-6)
                Q = Q * (3 ** (1 / 2))
            elif qk_norm_type == "l2x4":
                Q = nn.functional.normalize(Q, eps=1e-6)
                Q = Q * 2
            elif qk_norm_type == "dk":
                Q = Q / (dk ** (1 / 4))
            else:
                sys.exit("Unsupported QK-normalization")
            return Q

        self.norm_func = norm_func

    def forward(self, x):
        return self.norm_func(x)

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

        self.proj_q = self.conv_op(in_channels=self.input_channels_with_enc,
                                   out_channels=self.dk, kernel_size=self.projection_kernel_size,
                                   padding=self.projection_kernel_size // 2, bias=True)
        self.proj_k = self.conv_op(in_channels=self.input_channels_with_enc,
                                   out_channels=self.dk, kernel_size=self.projection_kernel_size,
                                   padding=self.projection_kernel_size // 2, bias=True)
        self.proj_v = self.conv_op(in_channels=self.input_channels, out_channels=self.dv,
                                   kernel_size=self.projection_kernel_size,
                                   padding=self.projection_kernel_size // 2, bias=True)
        self.qk_norm = QKNormalize(qk_norm_type, dk)



    def forward(self, x, x_enc):
        query = self.proj_q(x_enc)
        key = self.proj_k(x_enc)
        value = self.proj_v(x)

        Q = torch.flatten(query, start_dim=2)  # dim: (b,dk,vox)
        K = torch.flatten(key, start_dim=2)  # dim: (b,dk,vox)
        V = torch.flatten(value, start_dim=2)  # dim: (b,dv,vox)

        Q = self.qk_norm(Q)
        K = self.qk_norm(K)

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
                 qk_norm_type: str = "l2",
                 nnd: bool = True,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
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
        self.nnd = nnd

        if dv is None:
            self.dv = input_channels // num_heads
        else:
            self.dv = dv

        if dk is None:
            self.dk = self.dv
        else:
            self.dk = dk

        self.nnd_op = DropoutNormNonlin(input_channels,
                                        norm_op, norm_op_kwargs,
                                        dropout_op, dropout_op_kwargs,
                                        nonlin, nonlin_kwargs)
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

        # nnd2
        if self.nnd:
            result = self.nnd_op(result)

        if self.residual:
            result = result + x
        # old nnd
        # if self.nnd:
        #     result = self.nnd_op(result)

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
                 qk_norm_type: str = "l2",
                 save_attention: bool = False,
                 nnd: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None
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
    mhsa_ic = MHSA_interconnect(encoder, (-1, -2), nnd=True)
    print(mhsa_ic)
    [print(name) for name, _ in mhsa_ic.named_children()]
