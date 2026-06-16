import torch
from torch import nn

# Parent abstract class
class _GlobalPoolNd(nn.Module):
    __constants__ = ['ndim']
    ndim: int

    def __init__(self) -> None:
        super().__init__()
        self.pool_dims = tuple(range(2, 2 + self.ndim))

# Average pool family
class _GlobalAvgPoolNd(_GlobalPoolNd):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.pool_dims)

class GlobalAvgPool1d(_GlobalAvgPoolNd):
    ndim = 1

class GlobalAvgPool2d(_GlobalAvgPoolNd):
    ndim = 2

class GlobalAvgPool3d(_GlobalAvgPoolNd):
    ndim = 3

# Max pool family
class _GlobalMaxPoolNd(_GlobalPoolNd):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=self.pool_dims)

class GlobalMaxPool1d(_GlobalMaxPoolNd):
    ndim = 1

class GlobalMaxPool2d(_GlobalMaxPoolNd):
    ndim = 2

class GlobalMaxPool3d(_GlobalMaxPoolNd):
    ndim = 3

# LP-pool family
class _GlobalLpPoolNd(_GlobalPoolNd):
    __constants__ = ['p']

    def __init__(self, p = 2.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().pow(self.p).mean(dim=self.pool_dims).pow(1 / self.p)

class GlobalLpPool1d(_GlobalLpPoolNd):
    ndim = 1

class GlobalLpPool2d(_GlobalLpPoolNd):
    ndim = 2

class GlobalLpPool3d(_GlobalLpPoolNd):
    ndim = 3

# Trainable LP-pool family
class _GlobalLpPoolTrainableNd(_GlobalPoolNd):
    def __init__(self, p_init = 2.0) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() + 1e-8).pow(self.p).mean(dim=self.pool_dims).pow(1 / self.p)

class GlobalLpPoolTrainable1d(_GlobalLpPoolTrainableNd):
    ndim = 1

class GlobalLpPoolTrainable2d(_GlobalLpPoolTrainableNd):
    ndim = 2

class GlobalLpPoolTrainable3d(_GlobalLpPoolTrainableNd):
    ndim = 3