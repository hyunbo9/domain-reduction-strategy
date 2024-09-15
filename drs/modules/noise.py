import torch
from torch import nn
from torch.nn import functional as F


class NoiseHandler(nn.Module):

    def __init__(self, transient: torch.Tensor, mu: float = 0.05, sigma: float = 0.01, lambda_noise: float = 1):
        super().__init__()

        max_val = transient.max().item()
        self.mu = mu * max_val
        self.sigma = sigma * max_val
        self.lambda_noise = lambda_noise
        self.weights = nn.Parameter(torch.zeros_like(transient))

    def forward(self):
        weights = F.sigmoid(self.weights)
        noise = torch.ones_like(weights) * self.mu
        bias = weights * self.sigma * self.lambda_noise
        noise = noise + bias

        return noise
