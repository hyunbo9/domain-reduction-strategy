from typing import Tuple

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import drs.cuda_ops.backend as _backend
from .params import HiddenToTransientParams

ADJACENT_VOXEL_OFFSETS = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

__all__ = ['hidden_points_to_transient']


class HiddenPointsToTransient(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, coords: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor,
                params: HiddenToTransientParams):
        T = params.output_shape[0]
        T_start = params.T_start if params.T_start > 0 else 0
        T_end = params.T_end if (params.T_end > 0) and (params.T_end <= T) else T

        coords = coords.contiguous().float()
        albedo = albedo.contiguous().float()
        normal = normal.contiguous().float()
        wall_coords = params.scan_coords.contiguous().float()
        wall_indices = params.scan_indices.int().contiguous().int()
        output = torch.zeros(params.output_shape, device=coords.device, dtype=coords.dtype)

        _backend.hidden_points_to_transient_forward_cuda(coords, albedo, normal, wall_coords, wall_indices,
                                                         output,
                                                         params.bin_length,
                                                         T_start, T_end,
                                                         params.is_confocal,
                                                         params.light_coord[0],
                                                         params.light_coord[1],
                                                         params.light_coord[2],
                                                         params.falloff_scale,
                                                         params.is_retroreflective,
                                                         params.light_scan_is_reversed)
        ctx.params = params
        ctx.time_range = (T_start, T_end)
        ctx.save_for_backward(coords, albedo, normal, wall_coords, wall_indices)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, top_grad):
        coords, albedo, normal, wall_coords, wall_indices = ctx.saved_tensors
        grad_albedo = torch.zeros_like(albedo)
        grad_normal = torch.zeros_like(normal)
        params = ctx.params
        T_start, T_end = ctx.time_range

        _backend.hidden_points_to_transient_backward_cuda(coords, albedo, normal, wall_coords, wall_indices,
                                                          top_grad,
                                                          grad_albedo, grad_normal,
                                                          params.bin_length,
                                                          T_start, T_end,
                                                          params.is_confocal,
                                                          params.light_coord[0],
                                                          params.light_coord[1],
                                                          params.light_coord[2],
                                                          params.falloff_scale,
                                                          params.is_retroreflective,
                                                          params.light_scan_is_reversed)

        return None, grad_albedo, grad_normal, None


hidden_points_to_transient = HiddenPointsToTransient.apply
