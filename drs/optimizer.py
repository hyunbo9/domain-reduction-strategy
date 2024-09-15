import time

from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from drs.data import BaseData


class NLOSOptimizer(nn.Module):

    def __init__(self,
                 data: BaseData,
                 grid: nn.Module,

                 lr: float = 1,
                 lambda_l1: float = 0.1,
                 num_steps: int = 1000,
                 reduction_interval: int = 50,
                 upsample_interval: int = 100,

                 target_resolution: int = 128,
                 ):
        super().__init__()

        self.lr = lr
        self.lambda_l1 = lambda_l1
        self.num_steps = num_steps
        self.reduction_interval = reduction_interval
        self.upsample_interval = upsample_interval
        self.target_resolution = target_resolution

        self.data = data

    def _create_optimizer(self):
        return Adam(self.grid.parameters(), lr=self.lr)

    def run(self):
        start_time = time.time()
        for step in tqdm(range(1, self.num_steps + 1)):
            self.step()
            if (step % self.reduction_interval) == 0:
                self.domain_reduction()
            if (step % self.upsample_interval) == 0:
                self.upsample()
        latency = time.time() - start_time
        print(f'Latency: {latency}')

    def step(self):
        pass

    def _render_transient(self):
        coords = None
        albedo, normal = self.grid(coords)
        ## TODO: render transient
        return _hidden_points_to_transient(coords, albedo, normal, self.data.scan_coords, self.data.scan_indices)

    def domain_reduction(self):
        pass


def _hidden_points_to_transient(coords, albedo, normal, wall_coords, wall_indices):
    pass
