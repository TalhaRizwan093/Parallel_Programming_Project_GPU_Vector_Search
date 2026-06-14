cmake_minimum_required(VERSION 3.22)
project(parallel_search LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Default: RTX 3070 Mobile is sm_86 (Ampere consumer).
# Override on the command line for other GPUs:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 4090
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80   # A100
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90   # H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Useful flags for development:
#   --ptxas-options=-v  prints per-kernel register usage and shared-mem use
#   -lineinfo           lets Nsight Compute map metrics to source lines
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include_directories(${CMAKE_SOURCE_DIR}/engine/include)

# All three kernel translation units compile into one library
add_library(search_kernels
    engine/kernels/v1_naive.cu
    engine/kernels/v2_tiled.cu
    engine/kernels/v3_pq.cu
)
set_target_properties(search_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Driver / harness binary
add_executable(search engine/main.cu)
target_link_libraries(search PRIVATE search_kernels)
set_target_properties(search PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
