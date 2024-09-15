import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage._filters import _gaussian_kernel1d


class GaussianKernel3d(nn.Module):

    def __init__(self, sigma: float = 3.0, truncate: float = 4.0, order: float = 0.):
        super().__init__()

        sd = float(sigma)
        # make the radius of the filter equal to truncate standard deviations
        lw = int(truncate * sd + 0.5)

        kernel = _gaussian_kernel1d(sigma, order, lw)[::-1]
        self.register_buffer('kernel', torch.from_numpy(kernel.copy()).float())
        pass

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        K = self.kernel.size(0)
        padding = (K - 1) // 2

        for dim in range(3):
            weight_size = [1, 1, 1, 1, 1]
            padding_size = [0, 0, 0]
            weight_size[dim + 2] = K
            padding_size[dim] = padding
            weight = self.kernel.view(*weight_size)
            x = F.conv3d(x, weight, padding=tuple(padding_size))
            pass

        x = x.squeeze(0).squeeze(0)
        return x
