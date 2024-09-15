import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import drs.cuda_ops.backend as _backend

ADJACENT_VOXEL_OFFSETS = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelize(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, voxels, input_coords):
        input_coords = input_coords.contiguous()
        voxels = voxels.contiguous()

        Z, X, Y, C = voxels.size()
        voxel_coords, weights = _compute_voxel_coords_and_weights(input_coords, Z, X, Y)

        indices = _voxel_coords_to_indices(voxel_coords, Z, X, Y).int().contiguous()
        weights = weights.float().contiguous()
        outputs = torch.zeros(input_coords.size(0), C, device=voxels.device, dtype=voxels.dtype)

        _backend.trilinear_devoxelize_forward_cuda(voxels.view(-1, C), weights, indices, outputs)

        ctx.save_for_backward(indices, weights)
        ctx.size = (Z, X, Y)

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, point_grad):
        indices, weights = ctx.saved_tensors
        Z, X, Y = ctx.size
        C = point_grad.size(-1)
        point_grad = point_grad.contiguous()

        voxel_grad = torch.zeros(Z, X, Y, C, device=point_grad.device, dtype=point_grad.dtype)
        _backend.trilinear_devoxelize_backward_cuda(point_grad, weights, indices, voxel_grad)

        return voxel_grad, None


trilinear_devoxelize = TrilinearDevoxelize.apply


def _compute_voxel_coords_and_weights(input_coords, Z, X, Y):
    base_coords = torch.floor(input_coords).int()
    voxel_offsets = torch.tensor(ADJACENT_VOXEL_OFFSETS).unsqueeze(0).to(input_coords.device).int()
    all_coords = base_coords.unsqueeze(1) + voxel_offsets
    mask = (all_coords[..., 0] >= Z) | (all_coords[..., 1] >= X) | (all_coords[..., 2] >= Y)
    all_coords[..., 0].clamp_(0, Z - 1)
    all_coords[..., 1].clamp_(0, X - 1)
    all_coords[..., 2].clamp_(0, Y - 1)

    weights = torch.abs(input_coords.unsqueeze(1) - all_coords)
    weights = 1 - weights
    all_weights = weights[..., 0] * weights[..., 1] * weights[..., 2]
    all_weights[mask] = 0

    return all_coords.int().contiguous(), all_weights.contiguous()


def _voxel_coords_to_indices(voxel_coords, Z, X, Y):
    z, x, y = voxel_coords[..., 0], voxel_coords[..., 1], voxel_coords[..., 2]
    return (z * X * Y) + (x * Y) + y
