import os
from os import path
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import numpy as np

from drs.cuda_ops import HiddenToTransientParams

LIGHT_SPEED = 3e8


class BaseData(ABC):
    """
    Base class for handling data. This inherits the nn.Module class to utilize device management.
    """
    dataset_name: str = None

    wall_size: float = 1.
    bin_resolution: float = 0.1
    transient_rescale_factor: float = 10.
    subsample_stride: int = 1
    is_confocal: bool = True
    light_coord: Tuple[float, float, float] = (0., 0., 0.)
    is_retroreflective: bool = False
    light_scan_is_reversed: bool = False
    z_sample_ratio: float = 0.5

    def __init__(self, dataset_name: str, filename: str, raw_root_dir: str):
        """
        @param dataset_name: Name of the dataset
        @param raw_root_dir: Path to the raw data
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.raw_root_dir = raw_root_dir
        self.raw_filename = filename
        self.preprocessed_dir = path.join('datasets/preprocessed', self.dataset_name)

        self.transient = None
        self.scale_term = None
        self.shape = None
        self.scan_coords = None
        self.scan_indices = None

        self.T_start: int = -1
        self.T_end: int = -1

    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    def get_cuda_params(self):
        attributes = HiddenToTransientParams.attributes()
        params = {attr: getattr(self, attr) for attr in attributes if hasattr(self, attr)}
        params['output_shape'] = (self.T, self.N, self.N)
        return HiddenToTransientParams(**params)

    @property
    def T(self):
        return self.shape[0]

    @property
    def N(self):
        return self.shape[1]

    @property
    def Z(self):
        """
        Computes the z-axis resolution based on the bin resolution.
        """
        return int((2 * self.z_sample_ratio * self.wall_size) / self.bin_length)

    @property
    def bin_length(self):
        return self.bin_resolution * LIGHT_SPEED

    @property
    def raw_file_path(self):
        return path.join(self.raw_root_dir, self.raw_filename)

    @property
    def transient_file_path(self):
        save_filename = self.raw_filename
        if '.' in save_filename:
            save_filename = '.'.join(self.raw_filename.split('.')[:-1])
        return path.join(self.preprocessed_dir, f'{self.dataset_name}_{save_filename}.npz')

    def preprocess(self, force: bool = False):
        if not force and path.exists(self.transient_file_path):
            return

        transient = self.handle_raw_transient()
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        np.savez_compressed(self.transient_file_path, data=transient)

    @abstractmethod
    def handle_raw_transient(self):
        """
        Handle the raw transient data at 'raw_file_path' and return the processed transient data.
        """
        raise NotImplementedError

    def postprocess(self, transient: torch.Tensor):
        return transient

    def prepare(self):
        self.prepare_transient()
        self.prepare_scan_points()

    def prepare_transient(self):
        file = np.load(self.transient_file_path)
        transient = file['data']
        transient = torch.from_numpy(transient)
        if self.subsample_stride > 1:
            transient = transient[:, ::self.subsample_stride, ::self.subsample_stride]
        self.shape = (transient.size(0), transient.size(1))

        max_transient_value = transient.max().item()
        max_coords = torch.nonzero(transient == max_transient_value)[0]
        self.max_time_idx = max_coords[0].item()

        # rescaling (max normalize)
        scale_term = self.transient_rescale_factor / transient.max()
        transient = transient * scale_term
        transient = self.postprocess(transient)

        self.transient = transient
        self.scale_term = scale_term

    def prepare_scan_points(self):
        # create uniform scan coords
        N = self.shape[-1]
        grid_x, grid_y = torch.meshgrid(torch.arange(N), torch.arange(N))

        scan_indices = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).int()
        grid_z = torch.zeros_like(grid_x)
        scan_coords = torch.stack([grid_z, grid_x, grid_y], dim=-1).reshape(-1, 3).float()
        _scan_coords = (scan_coords[:, 1:] / N) - 0.5
        _scan_coords = _scan_coords * self.wall_size
        scan_coords[:, 1:] = _scan_coords

        self.scan_coords = scan_coords
        self.scan_indices = scan_indices
