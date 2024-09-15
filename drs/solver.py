import time
import os
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import cv2

import engine
from drs.cuda_ops import hidden_points_to_transient
from drs.data import BaseData
from drs.grid import ReconstructionGrid
from drs.sampling import InputPointsSampler
from drs.modules import GaussianKernel3d, NoiseHandler
from drs.loggers import BaseLogger


class NLOSOptimizationSolver(nn.Module):

    def __init__(self,
                 data: BaseData,
                 grid: ReconstructionGrid,
                 sampler: InputPointsSampler,
                 loggers: List[BaseLogger],

                 num_steps: int = 1000,
                 lr: float = 1,
                 lambda_l1: float = 0.1,
                 reduction_interval: int = 50,
                 upsample_interval: int = 100,
                 logging_interval: int = 50,

                 target_resolution: int = 128,
                 reduction_sigma: float = 3.0,
                 reduction_threshold: float = 0.05,
                 base_time_idx_scale: float = 0.25,
                 lambda_noise_handler: float = -1,
                 ):
        super().__init__()

        self.num_steps = num_steps
        self.lr = lr
        self.lambda_l1 = lambda_l1
        self.num_steps = num_steps
        self.reduction_interval = reduction_interval
        self.upsample_interval = upsample_interval
        self.logging_interval = logging_interval
        self.target_resolution = target_resolution
        self.reduction_sigma = reduction_sigma
        self.reduction_threshold = reduction_threshold
        self.bast_time_idx_scale = base_time_idx_scale

        self.loggers = loggers
        self.data = data
        self.grid = grid
        self.sampler = sampler
        self.gaussian_kernel = GaussianKernel3d(sigma=reduction_sigma)
        self.noise_handler = None
        if lambda_noise_handler > 0:
            self.noise_handler = NoiseHandler(data.transient, lambda_noise=lambda_noise_handler)

    def _create_optimizer(self):
        return Adam(self.grid.parameters(), lr=self.lr)

    def enable_debug_mode(self):
        self.num_steps = 1
        self.reduction_interval = 1
        self.upsample_interval = 1
        self.logging_interval = 1

    def run(self):
        self.sampler.set_active_domain(torch.ones_like(self.grid.albedo))
        print('---------- Optimization configuration ----------')
        print(f'  transient shape: {self.data.T}x{self.data.N}x{self.data.N}')
        print(f'  total steps: {self.num_steps}')
        Z, N = self.sampler.current_resolution
        print(f'  init domain resolution: {Z}x{N}x{N}')
        print(f'  init num. active domains: {self.sampler.num_active_domains}')
        print(f'  lr: {self.lr}')
        print(f'  lambda_l1: {self.lambda_l1}')
        print(f'  reduction interval: {self.reduction_interval}')
        print(f'  upsample interval: {self.upsample_interval}')
        print('------------------------------------------------')

        start_time = time.time()
        optimizer = self._create_optimizer()
        pbar = tqdm(range(1, self.num_steps + 1))
        for step in pbar:
            # main optimization logic
            optimizer.zero_grad()
            loss = self.step()
            loss.backward()
            optimizer.step()

            # logging, domain reduction, and upsampling
            should_log = (step % self.logging_interval) == 0
            should_reduce_domain = (step % self.reduction_interval) == 0
            should_upsample = (step % self.upsample_interval) == 0
            active_albedo = None
            if should_log or should_reduce_domain:
                active_albedo = self.grid.albedo * self.sampler.get_active_volume()
            if should_log:
                self.log_results(step, loss, active_albedo)
            if should_reduce_domain:
                self.domain_reduction(active_albedo)
            if should_upsample:
                self.upsample()
                optimizer = self._create_optimizer()
            pbar.set_postfix({'loss': loss.item(), 'a': f'{self.sampler.active_ratio * 100:.1f}%'})

        latency = time.time() - start_time
        print(f'Optimization finished. Latency: {latency:.2f} s')

    def step(self):
        grid_coords, coords = self.sampler.sample()
        albedo, normal = self.grid(grid_coords)

        base_time_idx = self.data.T * self.bast_time_idx_scale
        falloff_scale = base_time_idx * self.data.bin_length
        falloff_scale = 1 / falloff_scale
        params = self.data.get_cuda_params()
        params.falloff_scale = falloff_scale

        output = hidden_points_to_transient(coords, albedo, normal, params)
        delta = self.data.wall_size / self.sampler.current_resolution[-1]
        delta_Z = 1 / self.data.z_sample_ratio
        output = output * (delta ** 2) * delta_Z

        if self.noise_handler is not None:
            noise = self.noise_handler()
            output = output + noise
            output.clamp_min_(0)

        target = self.data.transient
        if self.data.T_start > 0:
            output = output[self.data.T_start:self.data.T_end]
            target = target[self.data.T_start:self.data.T_end]
        loss = F.mse_loss(output, target)

        if self.lambda_l1 > 0:
            Z, N = self.sampler.current_resolution
            l1_loss = torch.abs(albedo).sum() / (Z * N * N)
            loss = loss + self.lambda_l1 * l1_loss

        return loss

    def domain_reduction(self, active_albedo):
        active_volume = self.gaussian_kernel(active_albedo)
        threshold = self.reduction_threshold * active_volume.max().item()
        active_volume = active_volume > threshold
        self.sampler.set_active_domain(active_volume)

    def log_results(self, step, loss, active_albedo):
        active_normal = self.grid.normal * self.sampler.get_active_volume().unsqueeze(-1)
        max_indices = torch.max(active_albedo, dim=0)[1]
        for logger in self.loggers:
            logger.log_scalar('loss', step, loss.item())
            logger.log_result_volume('albedo', step, active_albedo, max_indices)
            logger.log_result_volume('normal', step, active_normal, max_indices)

    def upsample(self):
        if self.grid.albedo.size(-1) >= self.target_resolution:
            return

        self.grid.upsample()
        self.sampler.upsample()
