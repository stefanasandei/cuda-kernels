//
// Created by stefan on 8/31/25.
//

#ifndef CUDA_KERNELS_CUDA_TIMER_H
#define CUDA_KERNELS_CUDA_TIMER_H

#include <memory>

class CudaTimer
{
public:
  CudaTimer();
  ~CudaTimer();

  void start() const;
  float stop(); // returns elapsed ms

  float GetMS() const
  {
    return m_TimeMS;
  }

private:
  struct Impl; // has cuda specific members
  std::unique_ptr<Impl> pImpl_;

  float m_TimeMS;
};

#endif // CUDA_KERNELS_CUDA_TIMER_H
