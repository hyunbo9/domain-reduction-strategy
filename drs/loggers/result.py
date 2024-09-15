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

        if path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    def add_config(self, cfg):
        with open(path.join(self.save_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    def log_scalar(self, name, step, vals):
        # do nothing
        pass

    def log_result_volume(self, name, step, volume):
        name = name.replace('/', '_')
        save_dir = path.join(self.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        file_path = path.join(save_dir, f'{step}.png')

        img = torch.max(volume, dim=0)[0]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = (img * 255.).astype(np.uint8)
        cv2.imwrite(file_path, img)