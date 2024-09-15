#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#include "utils.h"
#include "hidden_to_transient.h"

template<typename scalar_t>
struct Vector3d {
    scalar_t z;
    scalar_t x;
    scalar_t y;
};

template<typename scalar_t>
struct DirectionalContext {
    bool is_out_of_bounds;
    int output_idx;
    scalar_t cosine_value;
    scalar_t r;
    Vector3d<scalar_t> direction;
};

template<typename scalar_t>
__device__ __forceinline__ Vector3d<scalar_t> create_vector(scalar_t z, scalar_t x, scalar_t y) {
    Vector3d<scalar_t> v;
    v.z = z;
    v.x = x;
    v.y = y;
    return v;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t get_norm(Vector3d<scalar_t> v) {
    return sqrt((v.z * v.z) + (v.x * v.x) + (v.y * v.y));
}

template<typename scalar_t>
__device__ __forceinline__ Vector3d<scalar_t> read_vector(const scalar_t *__restrict__ values) {
    return create_vector<scalar_t>(values[0], values[1], values[2]);
}

template<typename scalar_t>
__device__ __forceinline__ Vector3d<scalar_t> vector_diff(Vector3d<scalar_t> a, Vector3d<scalar_t> b) {
    return create_vector<scalar_t>(a.z - b.z, a.x - b.x, a.y - b.y);
}

template<typename scalar_t>
__device__ __forceinline__ DirectionalContext<scalar_t> compute_directional_context(const int idx_n, const int idx_m,
                                                                                    const ForwardParams params,
                                                                                    const scalar_t *__restrict__ albedo,
                                                                                    const scalar_t *__restrict__ normal,
                                                                                    const scalar_t *__restrict__ coords,
                                                                                    const scalar_t *__restrict__ wall_coords,
                                                                                    const int *__restrict__ wall_indices) {
    DirectionalContext<scalar_t> ctx;
    ctx.is_out_of_bounds = true;

    if ((idx_n >= params.N) || (idx_m >= params.M)) return ctx;

    auto coord = read_vector<scalar_t>(coords + idx_n * 3);
    auto normal_vec = read_vector<scalar_t>(normal + idx_n * 3);
    auto wall_coord = read_vector<scalar_t>(wall_coords + idx_m * 3);
    int x_idx = wall_indices[idx_m * 2 + 0];
    int y_idx = wall_indices[idx_m * 2 + 1];
    if ((x_idx >= params.X) || (x_idx < 0) || (y_idx >= params.Y) || (y_idx < 0)) return ctx;

    // computing t indices
    auto light_coord = params.is_confocal ?
        wall_coord :
        create_vector<scalar_t>(params.light_z, params.light_x, params.light_y);
    auto diff = vector_diff(wall_coord, coord);
    auto diff_in = vector_diff(coord, light_coord);

    float dist_in = get_norm(diff_in);
    float dist_out = get_norm(diff);
    float dist = dist_in + dist_out;
    float arrival_time = dist / params.bin_length;
    int t = round(arrival_time);
    if ((t < params.T_start) || (t >= params.T_end)) return ctx;

    float r_in = dist_in * params.falloff_scale;
    float r_out = dist_out * params.falloff_scale;
    float r = (r_in * r_in) * (r_out * r_out);

    // computing direction vector
    if (params.light_scan_is_reversed) {
        diff_in.z = -1 * diff.z;
        diff_in.x = -1 * diff.x;
        diff_in.y = -1 * diff.y;
    }
    auto direction = create_vector<scalar_t>(-1 * diff_in.z, -1 * diff_in.x, -1 * diff_in.y);
    scalar_t direction_norm = get_norm(direction);
    if (direction_norm <= 0) return ctx;

    direction.z /= direction_norm;
    direction.x /= direction_norm;
    direction.y /= direction_norm;

    // computing cosine value
    float cosine_value = 0;
    if (direction_norm > 0) {
        cosine_value = (direction.z * normal_vec.z) + (direction.x * normal_vec.x) + (direction.y * normal_vec.y);
    }
    if (params.is_retroreflective) {
        r = r_in * r_in;
        cosine_value = 1;
    }
    if (cosine_value <= 0) return ctx;

    int output_idx = (t * params.X * params.Y) + (x_idx * params.Y) + y_idx;

    ctx.is_out_of_bounds = false;
    ctx.output_idx = output_idx;
    ctx.cosine_value = cosine_value;
    ctx.r = r;
    ctx.direction = direction;
    return ctx;
}

template<typename scalar_t>
__global__ void hidden_points_to_transient_forward_kernel(const ForwardParams params,
                                                          const scalar_t *__restrict__ albedo,
                                                          const scalar_t *__restrict__ normal,
                                                          const scalar_t *__restrict__ coords,
                                                          const scalar_t *__restrict__ wall_coords,
                                                          const int *__restrict__ wall_indices,
                                                          scalar_t *__restrict__ output) {
    long global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx_n = global_idx / params.M;
    const int idx_m = global_idx % params.M;

    auto ctx = compute_directional_context<scalar_t>(idx_n, idx_m, params, albedo, normal, coords, wall_coords, wall_indices);
    if (ctx.is_out_of_bounds) return;

    atomicAdd(&output[ctx.output_idx], albedo[idx_n] * ctx.cosine_value / ctx.r);
}

template<typename scalar_t>
__global__ void hidden_points_to_transient_backward_kernel(const ForwardParams params,
                                                           const scalar_t *__restrict__ albedo,
                                                           const scalar_t *__restrict__ normal,
                                                           const scalar_t *__restrict__ coords,
                                                           const scalar_t *__restrict__ wall_coords,
                                                           const int *__restrict__ wall_indices,
                                                           const scalar_t *__restrict__ top_grad,
                                                           scalar_t *__restrict__ grad_albedo,
                                                           scalar_t *__restrict__ grad_normal) {
    long global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx_n = global_idx / params.M;
    const int idx_m = global_idx % params.M;

    auto ctx = compute_directional_context<scalar_t>(idx_n, idx_m, params, albedo, normal, coords, wall_coords, wall_indices);
    if (ctx.is_out_of_bounds) return;

    // we heuristically ignore the distance falloff term in the gradient computation, as it provides more stable optimization
    auto current_grad = top_grad[ctx.output_idx];
    atomicAdd(&grad_albedo[idx_n], current_grad * ctx.cosine_value);
    atomicAdd(&grad_normal[idx_n * 3 + 0], current_grad * albedo[idx_n] * ctx.direction.z);
    atomicAdd(&grad_normal[idx_n * 3 + 1], current_grad * albedo[idx_n] * ctx.direction.x);
    atomicAdd(&grad_normal[idx_n * 3 + 2], current_grad * albedo[idx_n] * ctx.direction.y);
}

void hidden_points_to_transient_forward_cuda(const at::Tensor coords,
                                             const at::Tensor albedo,
                                             const at::Tensor normal,
                                             const at::Tensor wall_coords,
                                             const at::Tensor wall_indices,
                                             at::Tensor output,
                                             float bin_length,
                                             int T_start, int T_end,
                                             bool is_confocal, float light_z, float light_x, float light_y,
                                             float falloff_scale,
                                             bool is_retroreflective,
                                             bool light_scan_is_reversed) {
    int N = coords.size(0);
    int M = wall_coords.size(0);
    int T = output.size(0);
    int X = output.size(1);
    int Y = output.size(2);

    ForwardParams params;
    params.N = N;
    params.M = M;
    params.T = T;
    params.X = X;
    params.Y = Y;
    params.bin_length = bin_length;
    params.T_start = T_start;
    params.T_end = T_end;
    params.is_confocal = is_confocal;
    params.light_z = light_z;
    params.light_x = light_x;
    params.light_y = light_y;
    params.falloff_scale = falloff_scale;
    params.is_retroreflective = is_retroreflective;
    params.light_scan_is_reversed = light_scan_is_reversed;

    CHECK_CUDA_TENSOR(coords);
    CHECK_CUDA_TENSOR(albedo);
    CHECK_CUDA_TENSOR(normal);
    CHECK_CUDA_TENSOR(wall_coords);
    CHECK_CUDA_TENSOR(wall_indices);
    CHECK_CUDA_TENSOR(output);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(coords.scalar_type(), "hidden_points_to_transient_forward_cuda", ([&] {
        hidden_points_to_transient_forward_kernel<scalar_t><<<NUM_BLOCKS(N * M), NUM_THREADS>>>(
            params,
            albedo.data_ptr<scalar_t>(),
            normal.data_ptr<scalar_t>(),
            coords.data_ptr<scalar_t>(),
            wall_coords.data_ptr<scalar_t>(),
            wall_indices.data_ptr<int>(),
            output.data_ptr<scalar_t>()
        );
    }));
    CHECK_CUDA_ERROR();
}

void hidden_points_to_transient_backward_cuda(const at::Tensor coords,
                                              const at::Tensor albedo,
                                              const at::Tensor normal,
                                              const at::Tensor wall_coords,
                                              const at::Tensor wall_indices,
                                              const at::Tensor top_grad,
                                              at::Tensor grad_albedo,
                                              at::Tensor grad_normal,
                                              float bin_length,
                                              int T_start, int T_end,
                                              bool is_confocal, float light_z, float light_x, float light_y,
                                              float falloff_scale,
                                              bool is_retroreflective,
                                              bool light_scan_is_reversed) {
    int N = coords.size(0);
    int M = wall_coords.size(0);
    int T = top_grad.size(0);
    int X = top_grad.size(1);
    int Y = top_grad.size(2);

    ForwardParams params;
    params.N = N;
    params.M = M;
    params.T = T;
    params.X = X;
    params.Y = Y;
    params.bin_length = bin_length;
    params.T_start = T_start;
    params.T_end = T_end;
    params.is_confocal = is_confocal;
    params.light_z = light_z;
    params.light_x = light_x;
    params.light_y = light_y;
    params.falloff_scale = falloff_scale;
    params.is_retroreflective = is_retroreflective;
    params.light_scan_is_reversed = light_scan_is_reversed;

    CHECK_CUDA_TENSOR(coords);
    CHECK_CUDA_TENSOR(albedo);
    CHECK_CUDA_TENSOR(normal);
    CHECK_CUDA_TENSOR(wall_coords);
    CHECK_CUDA_TENSOR(wall_indices);
    CHECK_CUDA_TENSOR(top_grad);
    CHECK_CUDA_TENSOR(grad_albedo);
    CHECK_CUDA_TENSOR(grad_normal);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(coords.scalar_type(), "hidden_points_to_transient_backward_cuda", ([&] {
        hidden_points_to_transient_backward_kernel<scalar_t><<<NUM_BLOCKS(N * M), NUM_THREADS>>>(
            params,
            albedo.data_ptr<scalar_t>(),
            normal.data_ptr<scalar_t>(),
            coords.data_ptr<scalar_t>(),
            wall_coords.data_ptr<scalar_t>(),
            wall_indices.data_ptr<int>(),
            top_grad.data_ptr<scalar_t>(),
            grad_albedo.data_ptr<scalar_t>(),
            grad_normal.data_ptr<scalar_t>()
        );
    }));
    CHECK_CUDA_ERROR();
}