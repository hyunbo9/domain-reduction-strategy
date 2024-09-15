from typing import Tuple
import math

import torch
import numpy as np

UPSAMPLE_OFFSETS = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]


class InputPointsSampler:

    def __init__(self,
                 hidden_size: float,
                 min_z: float = 0.1, sample_range: float = 0.5,
                 ):
        self.hidden_size = hidden_size
        self.min_z_ratio = min_z / hidden_size
        self.sample_range = sample_range
        self.current_resolution = None
        self.active_grid_indices = None

    @property
    def num_active_domains(self):
        return self.active_grid_indices.size(0)

    @property
    def active_ratio(self):
        return self.num_active_domains / (self.current_resolution[0] * self.current_resolution[1] ** 2)

    def set_active_domain(self, volume: torch.Tensor):
        min_z_idx = math.floor(self.min_z_ratio * volume.size(0))
        volume[:min_z_idx] = 0

        self.current_resolution = (volume.size(0), volume.size(1))
        self.active_grid_indices = torch.nonzero(volume).int()

    def get_active_volume(self):
        Z, N = self.current_resolution
        volume = torch.zeros(Z, N, N, dtype=torch.bool, device=self.active_grid_indices.device)
        volume[self.active_grid_indices[:, 0], self.active_grid_indices[:, 1], self.active_grid_indices[:, 2]] = 1
        return volume

    def sample(self):
        Z, N = self.current_resolution
        grid_indices = self.active_grid_indices.clone()

        grid_coords = grid_indices.clone().float()
        shift = torch.rand_like(grid_coords) - 0.5
        grid_coords += shift * 2 * self.sample_range
        grid_coords = grid_coords.view(-1, 3)
        grid_coords[:, 0].clamp_(0, Z - 1)
        grid_coords[:, 1:].clamp_(0, N - 1)

        coords = grid_coords.clone().float()
        coords[:, 0] = coords[:, 0] * self.hidden_size / Z
        coords[:, 1:] = ((coords[:, 1:] / N) - 0.5) * self.hidden_size

        return grid_coords, coords

    def upsample(self):
        offsets = torch.tensor(UPSAMPLE_OFFSETS, device=self.active_grid_indices.device)

        upsampled_indices = self.active_grid_indices.clone()
        upsampled_indices[..., 1:] *= 2
        upsampled_indices = upsampled_indices.unsqueeze(1) + offsets.unsqueeze(0)
        self.active_grid_indices = upsampled_indices.contiguous().view(-1, 3).contiguous()
        self.current_resolution = (self.current_resolution[0], self.current_resolution[1] * 2)
