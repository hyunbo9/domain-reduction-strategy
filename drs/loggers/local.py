import os
from os import path
import shutil

import torch
import numpy as np
import cv2
from omegaconf import OmegaConf

from .base import BaseLogger


class LocalLogger(BaseLogger):
    """
    Saves result images to the local disk.
    """

    def __init__(self, save_dir):
        super().__init__()

        self.save_dir = save_dir
        self.img_dir = path.join(self.save_dir, 'images')
        self.volume_dir = path.join(self.save_dir, 'volumes')

        if path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def add_config(self, cfg):
        with open(path.join(self.save_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    def init(self):
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.volume_dir, exist_ok=True)

    def log_scalar(self, name, step, vals):
        # do nothing
        pass

    def log_result_volume(self, name, step, volume, max_indices):
        name = name.replace('/', '_')

        img_path = path.join(self.img_dir, f'{step}_{name}.png')
        img = self.volume_to_img(volume, max_indices)
        cv2.imwrite(img_path, img)

        volume_path = path.join(self.volume_dir, f'{step}_{name}')
        np.save(volume_path, volume.detach().cpu().numpy())
