from os import path

import h5py
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from .base import BaseData


class StanfordData(BaseData):
    def __init__(self,
                 object_name: str,
                 raw_root_dir: str,
                 exposure: int = 180,
                 subsample_stride: int = 1,
                 ):
        filename = f'{object_name}_{exposure}min'
        super().__init__('stanford', filename, raw_root_dir)

        self.object_name = object_name
        self.filename = filename
        self.exposure = exposure

        ## handling metadata
        self.bin_resolution = 32e-12
        self.transient_rescale_factor = 10.
        self.wall_size = 2.
        self.subsample_stride = subsample_stride
        self.is_confocal = True

    def handle_raw_transient(self):
        object_dir = path.join(self.raw_root_dir, self.object_name)
        with h5py.File(path.join(object_dir, f'meas_{self.exposure}min.mat'), 'r') as file:
            transient = file['meas'][:]
        with h5py.File(path.join(object_dir, 'tof.mat'), 'r') as file:
            tof = file['tofgrid'][:]

        dt = self.bin_resolution * 1e12
        tof = np.floor(tof / dt).astype(np.int32)

        T, N = transient.shape[0], transient.shape[1]
        crop_size = 512
        cropped = np.zeros((crop_size, N, N), dtype=transient.dtype)
        for h, w in tqdm(list(np.ndindex(N, N)), desc='Stanford preprocessing'):
            start = int(tof[h, w])
            end = min(start + crop_size, T - 1)
            cropped_length = end - start
            cropped[:cropped_length, h, w] = transient[start:end, h, w]

        transient = cropped[:crop_size]
        transient = torch.from_numpy(transient).unsqueeze(0).unsqueeze(0)
        transient = F.avg_pool3d(transient, kernel_size=(1, 2, 2))
        transient = transient.squeeze(0).squeeze(0)
        transient = transient.permute(0, 2, 1)
        transient = transient.cpu().numpy()

        return transient

    def postprocess(self, transient):
        T, N, _ = transient.size()
        crop_length = int(T // 2)

        transient[:int(T * 0.15)] = 0
        intensity_sums = transient.reshape(1, 1, T, -1)
        intensity_sums = intensity_sums.sum(dim=-1)
        intensity_sums = F.avg_pool1d(intensity_sums, kernel_size=(crop_length), stride=1).squeeze(0).squeeze(0)

        shift = crop_length // 2
        start_index = crop_length // 2 + torch.argmax(intensity_sums) - shift
        end_index = start_index + crop_length

        transient[:start_index] = 0
        transient[end_index:] = 0
        self.T_start = start_index
        self.T_end = end_index

        return transient
