#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#include "utils.h"

template<typename scalar_t>
__global__ void trilinear_devoxelize_forward_cuda_kernel(int N, int C,
                                                         const scalar_t *__restrict__ voxels,
                                                         const scalar_t *__restrict__ weights,
                                                         const int *__restrict__ indices,
                                                         scalar_t *__restrict__ outputs) {
    long global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_n = global_idx / C;
    int idx_c = global_idx % C;

    if (idx_n >= N) return;

    const scalar_t *_weights = weights + (idx_n * 8);
    const int *_indices = indices + (idx_n * 8);
    scalar_t current_feat;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (_weights[k] == 0) continue;
        current_feat = voxels[_indices[k] * C + idx_c];
        outputs[idx_n * C + idx_c] += _weights[k] * current_feat;
    }
}

template<typename scalar_t>
__global__ void trilinear_devoxelize_backward_cuda_kernel(int N, int C,
                                                          const scalar_t *__restrict__ top_grad,
                                                          const scalar_t *__restrict__ weights,
                                                          const int *__restrict__ indices,
                                                          scalar_t *__restrict__ bottom_grad) {
    long global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_n = global_idx / C;
    int idx_c = global_idx % C;
    if (idx_n >= N) return;

    const scalar_t *_weights = weights + (idx_n * 8);
    const int *_indices = indices + (idx_n * 8);
    scalar_t current_grad = top_grad[idx_n * C + idx_c];

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (_weights[k] == 0) continue;
        atomicAdd(&bottom_grad[_indices[k] * C + idx_c], _weights[k] * current_grad);
    }
}

void trilinear_devoxelize_forward_cuda(const at::Tensor voxels,
                                       const at::Tensor weights,
                                       const at::Tensor indices,
                                       at::Tensor outputs) {
    int N = outputs.size(0);
    int C = outputs.size(1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(voxels.scalar_type(), "trilinear_devoxelize_forward_cuda", ([&] {
        trilinear_devoxelize_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS(N * C), NUM_THREADS, NUM_THREADS>>>(
            N, C,
            voxels.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            indices.data_ptr<int>(),
            outputs.data_ptr<scalar_t>()
        );
    }));
    CHECK_CUDA_ERROR();
}

void trilinear_devoxelize_backward_cuda(const at::Tensor top_grad,
                                        const at::Tensor weights,
                                        const at::Tensor indices,
                                        at::Tensor bottom_grad) {
    int N = top_grad.size(0);
    int C = top_grad.size(1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_grad.scalar_type(), "trilinear_devoxelize_backward_cuda", ([&] {
        trilinear_devoxelize_backward_cuda_kernel<scalar_t><<<NUM_BLOCKS(N * C), NUM_THREADS>>>(
            N, C,
            top_grad.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            indices.data_ptr<int>(),
            bottom_grad.data_ptr<scalar_t>()
        );
    }));
    CHECK_CUDA_ERROR();
}