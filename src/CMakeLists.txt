project(src)

file(GLOB_RECURSE KERNEL_HEADERS ${PROJECT_SOURCE_DIR}/*.cuh ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE KERNEL_SOURCES ${PROJECT_SOURCE_DIR}/*.cu)

add_library(cuda_kernels STATIC ${KERNEL_HEADERS} ${KERNEL_SOURCES})

set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_kernels common)
target_include_directories(cuda_kernels PUBLIC ./)
