//
// Created by stefan on 8/31/25.
//

#ifndef CUDA_KERNELS_COMMON_H
#define CUDA_KERNELS_COMMON_H

#include <cuda_runtime.h>

// RAII wrapper around cudaEvent_t
class CudaTimer
{
public:
  CudaTimer();
  ~CudaTimer();
  void start() const;
  [[nodiscard]] float stop() const; // returns elapsed ms
private:
  cudaEvent_t start_{}, stop_{};
};

// Convenience launcher
template <typename Kernel, typename... Args>
float timeKernel(
    Kernel k,
    dim3 grid,
    dim3 block,
    size_t shared = 0,
    cudaStream_t s = nullptr,
    Args&&... args)
{
  CudaTimer t;
  t.start();
  k<<<grid, block, shared, s>>>(std::forward<Args>(args)...);
  return t.stop();
}

#endif // CUDA_KERNELS_COMMON_H