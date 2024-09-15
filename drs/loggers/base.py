from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F
import numpy as np


class BaseLogger(ABC):

    @abstractmethod
    def add_config(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def log_scalar(self, name, step, vals):
        raise NotImplementedError

    @abstractmethod
    def log_result_volume(self, name, step, volume, max_indices):
        raise NotImplementedError

    def volume_to_img(self, volume, max_indices):
        max_indices = max_indices.unsqueeze(0)
        if volume.size(-1) == 3:
            img = torch.gather(volume, dim=0, index=max_indices.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(0)
            img = F.normalize(img, dim=-1, eps=1e-8)
            mask = img.abs().sum(dim=-1)
            img = (img + 1) / 2
            img[mask == 0] = 0
        else:
            img = torch.gather(volume, dim=0, index=max_indices).squeeze(0)
            img = img / img.max().item()

        img = img.detach().cpu().numpy()
        img = (img * 255.).astype(np.uint8)

        return img
