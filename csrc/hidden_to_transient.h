#ifndef _HIDDEN_TO_TRANSIENT_H
#define _HIDDEN_TO_TRANSIENT_H

#include <torch/torch.h>

/**
 * @param coords: input coordinates of shape (N, 3)
 * @param albedo: input albedo of shape (N, 1)
 * @param normal: input normal of shape (N, 3)
 * @param wall_coords: input wall coordinates of shape (M, 3)
 * @param wall_indices: input wall indices of shape (N, 3)
 * @param output: output tensor of shape (T, X, Y)
 * @param bin_length: length of each bin
 * @param T_start: start time index
 * @param T_end: end time index
 * @param use_light_pos: whether to use light position
 * @param light_z: light z position
 * @param light_x: light x position
 * @param light_y: light y position
 * @param max_time_idx: maximum time index
 * @param is_retroreflective: whether the surface has retroreflective BRDF
 * @param light_scan_is_reversed: whether the light scan is reversed (for some non-confocal measurements)
 */
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
                                             bool light_scan_is_reversed);

/**
 * @param coords: input coordinates of shape (N, 3)
 * @param albedo: input albedo of shape (N, 1)
 * @param normal: input normal of shape (N, 3)
 * @param wall_coords: input wall coordinates of shape (M, 3)
 * @param wall_indices: input wall indices of shape (N, 3)
 * @param top_grad: top gradient of shape (T, X, Y)
 * @param grad_albedo: gradient of albedo of shape (N, 1)
 * @param grad_normal: gradient of normal of shape (N, 3)
 * @param bin_length: length of each bin
 * @param T_start: start time index
 * @param T_end: end time index
 * @param use_light_pos: whether to use light position
 * @param light_z: light z position
 * @param light_x: light x position
 * @param light_y: light y position
 * @param max_time_idx: maximum time index
 * @param is_retroreflective: whether the surface has retroreflective BRDF
 * @param light_scan_is_reversed: whether the light scan is reversed (for some non-confocal measurements)
 */
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
                                              bool light_scan_is_reversed);

struct ForwardParams {
    int N;
    int M;
    int T;
    int X;
    int Y;
    float bin_length;
    int T_start;
    int T_end;
    bool is_confocal;
    float light_z;
    float light_x;
    float light_y;
    float falloff_scale;
    bool is_retroreflective;
    bool light_scan_is_reversed;
};

#endif