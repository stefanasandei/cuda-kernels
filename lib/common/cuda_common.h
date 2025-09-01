//
// Created by stefan on 9/1/25.
//

#ifndef CUDA_KERNELS_CUDA_COMMON_H
#define CUDA_KERNELS_CUDA_COMMON_H

#include "common/common.h"

#include <cuda_runtime.h>

template <typename Kernel, typename... Args>
void BenchmarkOrRun(
    BenchmarkOptional benchmarkOpt,
    Kernel k,
    dim3 grid,
    dim3 block,
    Args... args)
{
  if (benchmarkOpt != std::nullopt) {
    BenchmarkPayload& benchmark = benchmarkOpt.value().get();
    benchmark.OutTimes = std::make_unique<std::vector<float>>();

    for (int i = 0; i < benchmark.WarmupRuns; i++) {
      k<<<grid, block>>>(args...);
      cudaDeviceSynchronize();
    }

    for (int i = 0; i < benchmark.NumTrials; i++) {
      benchmark.Timer->start();
      k<<<grid, block>>>(args...);
      benchmark.Timer->stop();

      benchmark.OutTimes->push_back(benchmark.Timer->GetMS());
    }
  } else {
    k<<<grid, block>>>(args...);
    cudaDeviceSynchronize();
  }
}

#endif // CUDA_KERNELS_CUDA_COMMON_H
