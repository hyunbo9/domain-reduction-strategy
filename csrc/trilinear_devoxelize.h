#ifndef _TRILINEAR_H_
#define _TRILINEAR_H_

/**
 * @param voxels (L, C)
 * @param weights (N, 8)
 * @param indices (N, 8)
 * @param outputs (N, C)
 */
void trilinear_devoxelize_forward_cuda(const at::Tensor voxels,
                                       const at::Tensor weights,
                                       const at::Tensor indices,
                                       at::Tensor outputs);

/**
 * @param top_grad point-wise grad (N, C)
 * @param weights (N, 8)
 * @param indices (N, 8)
 * @param bottom_grad voxel grad (L, C)
 */
void trilinear_devoxelize_backward_cuda(const at::Tensor top_grad,
                                        const at::Tensor weights,
                                        const at::Tensor indices,
                                        at::Tensor bottom_grad);

#endif