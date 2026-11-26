from typing import Tuple
import torch
from torch import nn

from timm.layers import RotaryEmbeddingCat
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
    PatchDecode,
    PatchEmbed,
    PatchEmbed_deeper,
    PatchEmbedDeeperControlled, PatchEmbedDeeperControlledWithSkips, PatchDecodeWithSkips,
)
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.architectures.primus import Primus, PrimusV3, _PRIMUS_CONFIGS
from einops import rearrange

class UPrimus(Primus):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        num_classes: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            num_classes=num_classes,
            eva_depth=eva_depth,
            eva_numheads=eva_numheads,
            input_shape=input_shape,
            decoder_norm=decoder_norm,
            decoder_act=decoder_act,
            num_register_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_embed=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )
        self.keys_to_in_proj = (
            "down_projection.stem.blocks.0.conv1.conv",
            "down_projection.stem.blocks.0.conv1.all_modules.0",
        )
        self.down_projection = PatchEmbed_deeper(
            input_channels=input_channels,
            embed_dim=embed_dim,
            base_features=32,
            depth_per_level=(1, 1, 1),
            embed_proj_3x3x3=False,
            embed_block_style="residual",
            embed_block_type="basic",  # "basic" or "bottleneck" (if "residual" style)
        )

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # self.encoder = ResidualEncoder(
        #     input_channels,
        #     n_stages,
        #     features_per_stage,
        #     conv_op,
        #     kernel_sizes,
        #     strides,
        #     n_blocks_per_stage,
        #     conv_bias,
        #     norm_op,
        #     norm_op_kwargs,
        #     dropout_op,
        #     dropout_op_kwargs,
        #     nonlin,
        #     nonlin_kwargs,
        #     block,
        #     bottleneck_channels,
        #     global_enhancing=global_enhancing, global_enhancing_depth=global_enhancing_depth,
        #     global_enhancing_maxpool=global_enhancing_maxpool,
        #     global_enhancing_gate=global_enhancing_gate,
        #     return_skips=True,
        #     disable_default_stem=False,
        #     stem_channels=stem_channels,
        # )
        # self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.down_projection.apply(InitWeights_He(1e-2))

class PrimusP1(PrimusV3):
    def __init__(
            self,
            input_channels: int,
            embed_dim: int,
            patch_embed_size: Tuple[int, ...],
            num_classes: int,
            eva_depth: int = 24,
            eva_numheads: int = 16,
            input_shape: Tuple[int, ...] = None,
            decoder_norm=LayerNormNd,
            decoder_act=nn.GELU,
            num_register_tokens: int = 0,
            use_rot_pos_emb: bool = True,
            use_abs_pos_embed: bool = True,
            mlp_ratio=4 * 2 / 3,
            drop_path_rate=0.2,  #
            patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
            proj_drop_rate: float = 0.0,
            # drops out things related to the projection. That is in the MLP and at the end of EVA attention
            attn_drop_rate: float = 0.0,
            # drops attention, meaning connections between patches may bebroken up at random
            rope_impl=RotaryEmbeddingCat,
            rope_kwargs=None,
            init_values=0.1,  # Set to None to disable
            scale_attn_inner=True,
            depth_per_level: tuple[int, ...] = (1, 1, 1),
            ch_per_level: tuple[int, ...] = (32, 64, 256, 1024),
            add_skips: bool = True,
            nonlin=nn.LeakyReLU,
            skip_mode = "add", #"add" or "concat"
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            num_classes=num_classes,
            eva_depth=eva_depth,
            eva_numheads=eva_numheads,
            input_shape=input_shape,
            decoder_norm=decoder_norm,
            decoder_act=decoder_act,
            num_register_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_embed=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        self.down_projection = PatchEmbedDeeperControlledWithSkips(
            input_channels=input_channels,
            embed_dim=embed_dim,
            depth_per_level=depth_per_level,
            ch_per_level=ch_per_level,
            add_skips=add_skips,
            nonlin=nonlin,
        )
        self.up_projection = PatchDecodeWithSkips(
            patch_embed_size, embed_dim, num_classes, ch_per_level, norm=decoder_norm, activation=decoder_act, skip_mode=skip_mode,
        )

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))


    def forward(self, x, ret_mask=False):
        FW, FH, FD = x.shape[2:]  # Full W , ...
        x, skips = self.down_projection(x)
        # last output of the encoder is the input to EVA
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1]:]  # Removes the register tokens
        # In-fill in-active patches with empty tokens
        restored_x, restoration_mask = self.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        if restoration_mask is not None:
            mask = rearrange(restoration_mask, "b (w h d) -> b w h d", h=H, w=W, d=D)
            full_mask = (
                mask.repeat_interleave(FW // W, dim=1)
                .repeat_interleave(FH // H, dim=2)
                .repeat_interleave(FD // D, dim=3)
            )
            full_mask = full_mask[:, None, ...]  # Add channel dimension  # [B, 1, W, H, D]
        else:
            full_mask = None

        dec_out = self.up_projection(x, skips)
        if ret_mask:
            return dec_out, full_mask
        else:
            return dec_out


class PrimusP1X(PrimusP1):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        config_name: str,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0.2,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
        skip_mode = "add",
    ):
        conf = _PRIMUS_CONFIGS[config_name]
        super().__init__(
            input_channels=input_channels,
            num_classes=output_channels,
            embed_dim=conf["embed_dim"],
            eva_depth=conf["eva_depth"],
            eva_numheads=conf["eva_numheads"],
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
            skip_mode=skip_mode,
        )

class PrimusP1S(PrimusP1X):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0.2,
        patch_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
        skip_mode = "add",
    ):
        super().__init__(
            input_channels,
            output_channels,
            "S",
            patch_embed_size,
            input_shape,
            drop_path_rate,
            patch_drop_rate,
            rope_impl,
            rope_kwargs,
            init_values,
            scale_attn_inner,
            skip_mode
        )

class PrimusP1NS(PrimusP1X):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0.2,
        patch_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
        skip_mode = "add",
    ):
        super().__init__(
            input_channels,
            output_channels,
            "NS",
            patch_embed_size,
            input_shape,
            drop_path_rate,
            patch_drop_rate,
            rope_impl,
            rope_kwargs,
            init_values,
            scale_attn_inner,
            skip_mode
        )