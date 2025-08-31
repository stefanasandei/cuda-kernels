//
// Created by stefan on 8/31/25.
//

#include "cuda_timer.h"

#include <cuda_runtime.h>
#include <memory>

struct CudaTimer::Impl
{
  cudaEvent_t start_, stop_;
};

CudaTimer::CudaTimer() : pImpl_(std::make_unique<Impl>()), m_TimeMS(0)
{
  cudaEventCreate(&pImpl_->start_);
  cudaEventCreate(&pImpl_->stop_);
}

CudaTimer::~CudaTimer()
{
  cudaEventDestroy(pImpl_->start_);
  cudaEventDestroy(pImpl_->stop_);
}

void CudaTimer::start() const
{
  cudaEventRecord(pImpl_->start_);
}

float CudaTimer::stop()
{
  cudaEventRecord(pImpl_->stop_);
  cudaEventSynchronize(pImpl_->stop_);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, pImpl_->start_, pImpl_->stop_);

  m_TimeMS = ms;
  return ms;
}