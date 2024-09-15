from os import path

import h5py
import numpy as np
from tqdm import tqdm

from .base import BaseData, LIGHT_SPEED


class ZNLOSData(BaseData):
    def __init__(self,
                 filename: str,
                 raw_root_dir: str,
                 subsample_stride: int = 1,
                 ):
        super().__init__('znlos', filename, raw_root_dir)
        self.filename = filename

        with h5py.File(path.join(raw_root_dir, filename), 'r') as file:
            bin_length = file['deltaT'][()]
            bin_resolution = bin_length / LIGHT_SPEED

        ## handling metadata
        self.bin_resolution = bin_resolution
        self.transient_rescale_factor = 10.
        self.wall_size = 1.
        self.subsample_stride = subsample_stride
        self.is_confocal = filename.endswith('_conf.hdf5')

    def handle_raw_transient(self):
        file = h5py.File(self.raw_file_path, 'r')
        raw_transient = file['data'][:]
        shape = raw_transient.shape
        axes_are_inverted = False
        if len(shape) == 5:
            if shape[0] == 1:
                raw_transient = raw_transient[0]
            elif shape[-1] == 1:
                raw_transient = raw_transient[..., 0]
                raw_transient = np.transpose(raw_transient, (3, 2, 1, 0))
                axes_are_inverted = True

        raw_transient = np.sum(raw_transient, axis=1)  # using all bounce

        ## aligning transient
        T, N, N = raw_transient.shape
        wall_points = np.array(file['cameraGridPositions'])

        if axes_are_inverted:
            wall_points = np.transpose(wall_points, (2, 1, 0))
        camera_point = np.array(file['cameraPosition'])

        if len(camera_point.shape) == 1:
            camera_point = np.expand_dims(camera_point, -1)

        dists = wall_points - np.expand_dims(camera_point, -1)
        dists = np.sqrt(np.sum(dists ** 2, axis=0))
        align_indices = 2 * dists / self.bin_length

        # this aligning code causes a computational bottleneck
        transient = np.zeros_like(raw_transient)
        for y, x in tqdm(list(np.ndindex(N, N)), desc='ZNLOS preprocessing'):
            start = int(align_indices[y, x])
            transient[:T - start, y, x] = raw_transient[start:, y, x]
        return transient[:512]
