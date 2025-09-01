#include <common/common.h>
#include <common/cuda_common.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vectorAdd(int* a, int* b, int* c, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

void vectorAddHost(
    const int* h_a,
    const int* h_b,
    int* h_c,
    int size,
    BenchmarkOptional benchmarkOpt)
{
  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;

  cudaMalloc(&d_a, size * sizeof(int));
  cudaMalloc(&d_b, size * sizeof(int));
  cudaMalloc(&d_c, size * sizeof(int));

  cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  BenchmarkOrRun(
      benchmarkOpt,
      vectorAdd,
      blocksPerGrid,
      threadsPerBlock,

      d_a,
      d_b,
      d_c,
      size);

  cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
