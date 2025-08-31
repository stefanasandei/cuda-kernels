//
// Created by stefan on 8/31/25.
//

#ifndef CUDA_KERNELS_COMMON_H
#define CUDA_KERNELS_COMMON_H

#include <memory>

class CudaTimer
{
public:
  CudaTimer();
  ~CudaTimer();
  void start() const;
  [[nodiscard]] float stop() const; // returns elapsed ms
private:
  struct Impl; // has cuda specific members
  std::unique_ptr<Impl> pImpl_;
};

#endif // CUDA_KERNELS_COMMON_H