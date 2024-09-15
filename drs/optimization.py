import time

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from drs.cuda_ops import hidden_points_to_transient_loss
from drs.data import BaseData
from drs.grid import ReconstructionGrid
from drs.sampling import InputPointsSampler
from drs.gaussian import GaussianKernel3d


class NLOSOptimizationRunner(nn.Module):

    def __init__(self,
                 data: BaseData,
                 grid: ReconstructionGrid,
                 sampler: InputPointsSampler,

                 lr: float = 1,
                 lambda_l1: float = 0.1,
                 num_steps: int = 1000,
                 reduction_interval: int = 50,
                 upsample_interval: int = 100,
                 logging_interval: int = 50,

                 target_resolution: int = 128,

                 ## domain reduction params
                 reduction_sigma: float = 3.0,
                 reduction_threshold: float = 0.05
                 ):
        super().__init__()

        self.lr = lr
        self.lambda_l1 = lambda_l1
        self.num_steps = num_steps
        self.reduction_interval = reduction_interval
        self.upsample_interval = upsample_interval
        self.logging_interval = logging_interval
        self.target_resolution = target_resolution
        self.reduction_sigma = reduction_sigma
        self.reduction_threshold = reduction_threshold

        self.data = data
        self.grid = grid
        self.sampler = sampler
        self.gaussian_kernel = GaussianKernel3d(sigma=reduction_sigma)

    def _create_optimizer(self):
        return Adam(self.grid.parameters(), lr=self.lr)

    def run(self):
        self.sampler.set_active_domain(torch.ones_like(self.grid.albedo))

        start_time = time.time()
        optimizer = self._create_optimizer()
        for step in tqdm(range(1, self.num_steps + 1)):
            # main optimization logic
            optimizer.zero_grad()
            loss = self.step()
            loss.backward()
            optimizer.step()

            if (step % self.logging_interval) == 0:
                self.log_results()
            if (step % self.reduction_interval) == 0:
                self.domain_reduction()
            if (step % self.upsample_interval) == 0:
                self.upsample()
                optimizer = self._create_optimizer()

        latency = time.time() - start_time
        print(f'Latency: {latency}')

    def step(self):
        grid_coords, coords = self.sampler.sample()
        albedo, normal = self.grid(grid_coords)

        base_time_idx = self.data.max_time_idx * self.time_idx_scale
        falloff_scale = base_time_idx * self.data.bin_length
        falloff_scale = 1 / falloff_scale
        loss = hidden_points_to_transient_loss(coords, albedo, normal, self.data.scan_coords, self.data.scan_indices,
                                               self.data.transient, True,
                                               self.data.bin_length,
                                               self.data.T_start, self.data.T_end,
                                               self.data.is_confocal, self.data.light_coord,
                                               falloff_scale,
                                               self.data.is_retroreflective,
                                               self.data.light_scan_is_reversed)
        return loss.mean()

    def domain_reduction(self):
        active_volume = self.gaussian_kernel(self.grid.albedo)
        threshold = self.reduction_threshold * active_volume.max().item()
        active_volume = active_volume > threshold
        self.sampler.set_active_domain(active_volume)

    def log_results(self):
        pass

    def upsample(self):
        self.grid.upsample()
        self.sampler.upsample()
