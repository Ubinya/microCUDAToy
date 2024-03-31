#include <torch/extension.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

//#define DEBUG

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)


#ifdef DEBUG
#   define TIMER PerfTimer timer = PerfTimer()
#   define TIMER_CHECK(x) timer.check(x) 
#   define DEBUG_PRINT(x) std::cout << STRINGIFY(x) ":" << x << std::endl
#else
#   define TIMER
#   define TIMER_CHECK(x)
#   define DEBUG_PRINT(x)
#endif 

#define BLOCK_SIZE 256

namespace F = torch::nn::functional;
namespace I = torch::indexing;

class PerfTimer {

    cudaStream_t m_stream;
    std::chrono::time_point<std::chrono::system_clock> m_curr;

public:
    
    PerfTimer() {
        m_stream = at::cuda::getCurrentCUDAStream();    
        cudaStreamSynchronize(m_stream);
        m_curr = std::chrono::system_clock::now();
    }

    void check(std::string checkpoint) {
        cudaStreamSynchronize(m_stream);
        auto end = std::chrono::system_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end-m_curr);
        std::cout << checkpoint << ": " << elapsed_seconds.count() << " us" << std::endl;
        m_curr = end;
    }
};

template <typename scalar_t>
__global__ void kernel_BN(
    const int batch_size
    const scalar_t * __restrict__ src_data,
    scalar_t * dst_data
){
    const int idx = blockIdx.x*blockDim.x + threadIdx.x; //global idx
    if (idx > batch_size) return;

    auto value = src_data[idx];
}


__device__ void kernel_parallel_summing(
    const int batch_size
    const float * __restrict__ src_data,
    float * dst_data
){
    __shared__ float partialSum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (index < N) {
        partialSum[tid] = input[index];
    } else {
        partialSum[tid] = 0.0f;
    }
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads(); // Synchronize threads within the block
    }

    // Store the partial sum for this block in global memory
    if (tid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
    
}

__global__ void kernel_partion_summing(const float *input, float *output) {
    
}




//cpp Launchder function
//at::Tensor myBatchNormLauncher(input, batch, channels, height, width);
