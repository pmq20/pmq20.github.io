#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstring>
#include <ctime>

#define WARP_SIZE 32

clock_t start,end;

__global__ void
_cuda_parallel_sum(int *in, int num_elements, int *sum)
{
    __syncthreads();
    __shared__ int buffer[WARP_SIZE];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int temp;
    while(globalIdx < num_elements)
    {
        temp = in[globalIdx];
        for (int delta = WARP_SIZE/2; delta > 0; delta /= 2)
        {
             temp+= __shfl_xor(temp, delta);
        }
        if (lane == 0)
        {
            buffer[threadIdx.x / WARP_SIZE] = temp;
        }
        __syncthreads();
        if(threadIdx.x < WARP_SIZE) 
        {
            temp = buffer[threadIdx.x];
            for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
            {  
                temp += __shfl_xor(temp, delta);
            }
        }
        if(threadIdx.x == 0)
        {
            atomicAdd(sum, temp);
        }
        globalIdx += blockDim.x * gridDim.x;
        __syncthreads();
    }
}

int cuda_parallel_sum(int * a, int N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_SMs = prop.multiProcessorCount;
    start = std::clock();
    int batch_size = num_SMs * 1024;
    int padding = (batch_size - (N % batch_size)) % batch_size;
    int * b = new int[N + padding];
    memcpy(b, a, N * sizeof(int));
    memset(b + N, 0, padding * sizeof(int));
    int *d_b;
    cudaMalloc( (void**) &d_b, (N + padding) * sizeof(int) );
    cudaMemcpy( d_b, b, (N + padding) * sizeof(int), cudaMemcpyHostToDevice );
    int result = 0.0;
    int * d_result;
    cudaMalloc( (void**) &d_result, sizeof(int) );
    cudaMemcpy( d_result, &result, sizeof(int), cudaMemcpyHostToDevice );
    _cuda_parallel_sum<<< num_SMs, 1024 >>>(d_b, N + padding, d_result);
    cudaMemcpy( &result, d_result, sizeof(int), cudaMemcpyDeviceToHost );
    end = std::clock();
    cudaFree(d_result);
    cudaFree(d_b);
    free(b);
    return result;
}


int cpu_sum(int * a, int N) {
	int sum = 0;
	start = std::clock();
	for(int i = 0; i < N; ++i) {
    sum += a[i];
  }
  end = std::clock();
  return sum;
}

int main() {
  const int N = 10000;
  cudaSetDevice(0);
  int * a = new int[N];
  for(int i = 0; i <= N; i++) {
    a[i] = i;
  }
  int result = cuda_parallel_sum(a, N);
  printf("GPU sum: %d - total time: %lfms\n", result, 1000.0 * (end-start) / CLOCKS_PER_SEC);

  for(int i = 0; i <= N; ++i) {
    a[i] = i;
  }
  result = cpu_sum(a, N);
  printf("CPU sum: %d - total time: %lfms\n", result, 1000.0 * (end-start) / CLOCKS_PER_SEC);
  return 0;
}
