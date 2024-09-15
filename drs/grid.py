import torch
from torch import nn
from torch.nn import functional as F

from drs.cuda_ops import trilinear_devoxelize


class ReconstructionGrid(nn.Module):

    def __init__(self,
                 init_resolution: tuple,
                 init_scale: float = 1.,
                 upsample_rescale_factor: float = 1,
                 ):
        super().__init__()

        self.init_resolution = init_resolution
        self.init_scale = init_scale
        self.upsample_rescale_factor = upsample_rescale_factor

        Z, N = self.init_resolution
        self.albedo = nn.Parameter(torch.rand((Z, N, N)) * self.init_scale)
        self.normal = nn.Parameter(torch.zeros((Z, N, N, 3)))
        self.register_buffer('base_normal', torch.tensor([-1, 0, 0]))

    def forward(self, coords):
        coords = coords.float().contiguous()
        albedo = trilinear_devoxelize(self.albedo.unsqueeze(-1), coords).squeeze(-1)
        normal = trilinear_devoxelize(self.normal, coords)

        albedo = F.elu(albedo)
        normal = F.tanh(normal) + self.base_normal
        normal = F.normalize(normal)

        return albedo, normal

    def upsample(self):
        Z, N = self.albedo.size(0), self.albedo.size(1)

        albedo = self.albedo.unsqueeze(0).unsqueeze(0)
        normal = self.normal.permute(3, 0,  1, 2).unsqueeze(0)
        albedo = F.interpolate(albedo, size=(Z, N * 2, N * 2), mode='trilinear').squeeze(0).squeeze(0)
        normal = F.interpolate(normal, size=(Z, N * 2, N * 2), mode='trilinear').squeeze(0).permute(1, 2, 3, 0)
        
        self.albedo = nn.Parameter(albedo)
        self.normal = nn.Parameter(normal)
