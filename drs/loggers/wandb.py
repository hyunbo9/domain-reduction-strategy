import time
import os

import wandb
from omegaconf import OmegaConf

from .base import BaseLogger


class WandbLogger(BaseLogger):

    def __init__(self, name, save_dir, config_path):
        super().__init__()

        cfg = self._load_and_validate_config(config_path)
        self.name = name
        self.save_dir = save_dir
        self.project = cfg.project
        self.entity = cfg.entity
        self.run = None
        self.config = None

        os.environ['WANDB_API_KEY'] = cfg.api_key
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_and_validate_config(self, config_path):
        cfg = OmegaConf.load(config_path)
        assert 'project' in cfg
        assert 'entity' in cfg
        assert 'api_key' in cfg
        return cfg

    def add_config(self, config):
        self.config = config

    def init(self):
        run_id = f'{int(time.time())}_{self.name}'
        self.run = wandb.init(project=self.project,
                              entity=self.entity,
                              id=run_id,
                              name=self.name,
                              config=self.config,
                              dir=self.save_dir,
                              )

    def log_scalar(self, name, step, vals):
        self.run.log({name: vals}, step=step)

    def log_result_volume(self, name, step, volume, max_indices):
        img = self.volume_to_img(volume, max_indices)
        img = wandb.Image(img)
        self.run.log({name: img}, step=step)
