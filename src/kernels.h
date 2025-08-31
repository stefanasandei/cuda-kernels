//
// Created by stefan on 8/31/25.
//

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

void vectorAddHost(
    const int* h_a,
    const int* h_b,
    int* h_c,
    int size,
    const std::unique_ptr<CudaTimer>& timer);

#endif
