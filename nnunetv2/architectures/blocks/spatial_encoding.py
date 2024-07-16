from typing import Tuple

import torch
import torch.nn as nn

def get_sincos_embeding(
    grid_size: Tuple[int], embed_dim: int = 96, min_freq: float = 1/32.0, desync_coeff: float = 5, device: torch.device = None
) -> torch.nn.Parameter:
    """
    Builds a sin-cos position embedding based on the given grid size, embed dimension, minimal frequency.
    Modified after: https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/networks/blocks/pos_embed_utils.py
    Default values are hand-picked based on dot-product as a distance metric between two positional encodings. The
    values guarantee monotonous distance function for 1, 2, and 3 dimensions.

    Args:
        grid_size (Tuple[int]): The size of the grid in each spatial dimension.
        embed_dim (int): The dimension of the embedding.
        min_freq (float): Minimal frequency of the sin-cos position embedding.
        desync_coeff (float): Controls speed of correlation decay.

    Returns:
        pos_embed (nn.Parameter): The sin-cos position embedding as a fixed parameter.
    """
    spatial_dims = len(grid_size)
    max_dim = max(grid_size)

    grid_list = [torch.arange(i, dtype=torch.float32, device=device) / max_dim * desync_coeff for i in grid_size]
    grid_list = torch.meshgrid(grid_list, indexing="ij")

    pos_dim = embed_dim // (spatial_dims * 2)
    f_index = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    freq = min_freq ** f_index

    arg_list = [torch.einsum("d,...->d...", [freq, grid_i]) for grid_i in grid_list]
    enc_stack = [f(arg) for arg in arg_list for f in (torch.sin, torch.cos)]
    pos_emb = torch.cat(enc_stack, dim=0)[None, :]

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False

    return pos_embed

# def test(grid_size: Tuple[int], embed_dim: int, min_freq: float = 1/32.0, desync_coeff: float = 4 * torch.pi):
#     em = get_sincos_embeding(grid_size, embed_dim, min_freq, desync_coeff)
#     dist = torch.einsum("nci,ncx->ix", [em, em]) / em.shape[1] * 2
#     diag = dist.flip(0).diag()
#     print(torch.min(diag), "at", torch.argmin(diag))
#     import matplotlib.pyplot as plt
#     plt.plot(diag)
#     plt.show(block=True)
