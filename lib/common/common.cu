//
// Created by stefan on 8/31/25.
//

#include "common.h"

#include <stdexcept>

CudaTimer::CudaTimer()
{
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}
CudaTimer::~CudaTimer()
{
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

void CudaTimer::start()
{
  cudaEventRecord(start_);
}
float CudaTimer::stop()
{
  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start_, stop_);
  return ms;
}