#include <torch/extension.h>

#include "hidden_to_transient.h"
#include "trilinear_devoxelize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hidden_points_to_transient_forward_cuda", &hidden_points_to_transient_forward_cuda, "hidden to transient forward");
    m.def("hidden_points_to_transient_backward_cuda", &hidden_points_to_transient_backward_cuda, "hidden to transient backward");
    m.def("trilinear_devoxelize_forward_cuda", &trilinear_devoxelize_forward_cuda, "trilinear devoxelize forward");
    m.def("trilinear_devoxelize_backward_cuda", &trilinear_devoxelize_backward_cuda, "trilinear devoxelize backward");
}