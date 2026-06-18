from typing import Tuple, Union, Type, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.linear import Linear

import nnunetv2.architectures.operations.global_pooling
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from nnunetv2.architectures.operations.combo_operations import LinearNormNonlinDropout
from nnunetv2.architectures.operations.global_pooling import _GlobalPoolNd


class AuxMLPHead(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 active_stages: Union[int, List[int], Tuple[int, ...]],
                 hidden_features: Union[int, List[int], Tuple[int, ...]],
                 output_logits: int,
                 pool_op: Type[_GlobalPoolNd],
                 pool_op_kwargs: dict = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 final_act: Union[None, Type[nn.Module]] = None,
                 final_act_kwargs: dict = None,
                 block_grad: bool = False,
                 ):
        super(AuxMLPHead, self).__init__()
        if isinstance(active_stages, int):
            active_stages = (active_stages,)
        if isinstance(active_stages, list):
            active_stages = tuple(active_stages)
        if isinstance(hidden_features, int):
            hidden_features = (hidden_features,)
        if isinstance(hidden_features, list):
            hidden_features = tuple(hidden_features)

        pool_kwargs = {} if pool_op_kwargs is None else pool_op_kwargs
        final_act_kwargs = {} if final_act_kwargs is None else final_act_kwargs

        self.features_per_stage = encoder.output_channels
        self.active_stages = active_stages
        self.hidden_features = hidden_features
        self.block_grad = block_grad

        self.total_input_features = sum((self.features_per_stage[i] for i in active_stages))
        self.input_features = (self.total_input_features,) +  self.hidden_features[:-1]

        self.pool_ops = nn.ModuleList([pool_op(**pool_kwargs) for stage in active_stages])
        self.fc_ops = [
            LinearNormNonlinDropout(in_features, out_features,
                                    norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs,
                                    nonlin, nonlin_kwargs)
            for in_features, out_features in zip(self.input_features, hidden_features)]

        self.final_ops = []
        self.final_ops.append(Linear(hidden_features[-1], output_logits, bias=True))
        if final_act is not None:
            self.final_ops.append(final_act(**final_act_kwargs))

        self.mlp = nn.Sequential(*self.fc_ops, *self.final_ops)

    def forward(self, skips):
        if self.block_grad:
            skips = [skip.detach() for skip in skips]
        feature_list = [self.pool_ops[i_pool](skips[i_stage]) for i_pool, i_stage in enumerate(self.active_stages)]
        feature_vector = torch.cat(feature_list, 1)
        return self.mlp(feature_vector)


    def compute_memory(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # go over active stages and sum up the memory
        output = np.int64(0)
        # calculation missing! Probably not important though...
        return output

if __name__ == '__main__':
    data = torch.rand((3, 2, 64, 32, 32))

    encoder = ResidualEncoder(2, 6, (32, 64, 128, 256, 320, 320), nn.Conv3d, 3,
                              ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                              (1, 3, 4, 6, 6, 6), True, nn.modules.instancenorm.InstanceNorm3d,
                              nonlin=nn.ReLU,
                              return_skips=True, disable_default_stem=False, stem_channels=None)

    aux_head = AuxMLPHead(encoder = encoder, active_stages=[-3,-1], hidden_features=(16,64), output_logits=2,
                          pool_op=nnunetv2.architectures.operations.global_pooling.GlobalLpPoolTrainable3d, pool_op_kwargs={'p': 3},
                          norm_op=nn.BatchNorm1d, norm_op_kwargs=None,
                          dropout_op=nn.Dropout, dropout_op_kwargs={'p': 0.1, 'inplace': True},
                          nonlin=nn.ReLU, nonlin_kwargs=None,
                          final_act=nn.Sigmoid,
                          block_grad=True)

    #print(encoder)
    print(aux_head)
    [print(name) for name, _ in aux_head.named_children()]

    out = aux_head(encoder(data))
    print(out)