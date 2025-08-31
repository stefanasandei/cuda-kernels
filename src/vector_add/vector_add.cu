#define ARR_LEN(x) (sizeof(x) / sizeof(int))

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include <common.h>

__global__ void vectorAdd(int* a, int* b, int* c)
{
  int idx = threadIdx.x;

  c[idx] = a[idx] + b[idx];
}

int vec()
{
  int a[] = {1, 2, 3};
  int b[] = {4, 5, 6};
  int c[ARR_LEN(a)] = {0};

  int* cudaA = nullptr;
  int* cudaB = nullptr;
  int* cudaC = nullptr;

  cudaMalloc(&cudaA, sizeof(a));
  cudaMalloc(&cudaB, sizeof(b));
  cudaMalloc(&cudaC, sizeof(c));

  cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaC, c, sizeof(c), cudaMemcpyHostToDevice);

  vectorAdd<<<1, ARR_LEN(c)>>>(cudaA, cudaB, cudaC);

  cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

  for (const int i : c) {
    printf("%d ", i);
  }
  printf("\n");

  return 0;
}
