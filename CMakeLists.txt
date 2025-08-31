cmake_minimum_required(VERSION 4.0)
project(cuda_kernels CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)
