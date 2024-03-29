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


__global__ void kernel_sum_mat(
    const float* __restrict__ mat_a,
    const float* __restrict__ mat_b,
    float* __restrict__ mat_out,
    const int n
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x; //global idx
    int stride = blockDim.x*gridDim.x; //block size * block num
    if (idx > n) return;
    
    for (int i=idx; i<n; i+=stride) {
        mat_out[i] = mat_a[i] + mat_b[i];
    }
}

//cpp Launchder function
at::Tensor myBatchNormLauncher(input, batch, channels, height, width);
