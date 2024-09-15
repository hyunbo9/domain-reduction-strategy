#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256

#define NUM_BLOCKS(work_size) ((work_size) + NUM_THREADS - 1) / NUM_THREADS


// error checking code borrowed from "https://github.com/mit-han-lab/pvcnn"
#define CHECK_CUDA_ERROR() \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",           \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__,          \
              __FILE__);                                                       \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
                             TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#endif