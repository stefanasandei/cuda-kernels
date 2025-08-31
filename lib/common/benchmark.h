//
// Created by stefan on 8/31/25.
//

#ifndef CUDA_KERNELS_BENCHMARK_H
#define CUDA_KERNELS_BENCHMARK_H

#include <vector>

float GetMean(const std::vector<float>& values);
float GetStd(const std::vector<float>& values);

#endif // CUDA_KERNELS_BENCHMARK_H
