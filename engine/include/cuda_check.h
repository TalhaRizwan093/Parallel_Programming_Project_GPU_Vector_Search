#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                                     \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess) {                                                    \
        std::fprintf(stderr,                                                      \
            "[CUDA ERROR] %s:%d  %s -> %s\n",                                     \
            __FILE__, __LINE__, #expr, cudaGetErrorString(_err));                 \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

#define CUDA_LAUNCH_CHECK(name) do {                                              \
    cudaError_t _err = cudaGetLastError();                                        \
    if (_err != cudaSuccess) {                                                    \
        std::fprintf(stderr,                                                      \
            "[CUDA LAUNCH ERROR] %s -> %s\n",                                     \
            (name), cudaGetErrorString(_err));                                    \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)
