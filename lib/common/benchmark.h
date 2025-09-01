//
// Created by stefan on 8/31/25.
//

#ifndef CUDA_KERNELS_BENCHMARK_H
#define CUDA_KERNELS_BENCHMARK_H

#include <memory>
#include <vector>

#include "cuda_timer.h"

struct BenchmarkPayload
{
  std::unique_ptr<CudaTimer> Timer;
  int WarmupRuns, NumTrials;
  std::unique_ptr<std::vector<float>> OutTimes;
};

float GetMean(const std::vector<float>& values);
float GetStd(const std::vector<float>& values);

#endif // CUDA_KERNELS_BENCHMARK_H
