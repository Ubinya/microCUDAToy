#include <torch/extension.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <math.h>

//#define DEBUG

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

#define CP_D2H(dst, src, size) cudaMemcpy((dst), (src), (size), cudaMemcpyDeviceToHost)
#define CP_H2D(dst, src, size) cudaMemcpy((dst), (src), (size), cudaMemcpyHostToDevice)



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
#define MAX_GRID_SIZE 512
#define EPSILON 1e-5


__global__ void kernel_clear_partialArray(
    float * scr_data
){
    scr_data[threadIdx.x] = 0.0f;
}

__global__ void kernel_parallel_sum(
    const int batch_size,
    const float * __restrict__ src_data,
    float * dst_data
){
    __shared__ float partialArr[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (index < batch_size) {
        partialArr[tid] = src_data[index];
    } else {
        partialArr[tid] = 0.0f;
    }
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialArr[tid] += partialArr[tid + stride];
        }
        __syncthreads();
    }

    // Store the partial sum for this block to global memory
    if (tid == 0) {
        dst_data[blockIdx.x] = partialArr[0];
    }
}

__global__ void kernel_sum_array(
    float *input, 
    float *output
){

    __shared__ float partialArr[MAX_GRID_SIZE];
    int tid = threadIdx.x;
    // Load data into shared memory
    partialArr[tid] = input[tid];
    __syncthreads(); 
    
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialArr[tid] += partialArr[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = partialArr[0];
    }
}

__global__ void kernel_parallel_variance(
    const int batch_size,
    const float * mean,
    const float * __restrict__ src_data,
    float * dst_data
){
    __shared__ float partialArr[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (index < batch_size) {
        partialArr[tid] = pow((src_data[index] - *mean), 2);
    } else {
        partialArr[tid] = 0.0f;
    }
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialArr[tid] += partialArr[tid + stride];
        }
        __syncthreads();
    }

    // Store the partial sum for this block to global memory
    if (tid == 0) {
        dst_data[blockIdx.x] = partialArr[0];
    }
}

__global__ void kernel_normalize(
    const int batch_size,
    const float * mean,
    const float * var,
    const float * __restrict__ src_data,
    float * dst_data,
    const float gamma,
    const float beta
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size) {
        dst_data[index] = gamma * (src_data[index] - *mean) / sqrt(*var + EPSILON) + beta;
    }
}

int round_up_to_power_of_two(int n) {
    if (n && !(n & (n - 1))) {
        // If n is already a power of 2, return it
        return n;
    } else {
        // If n is not a power of 2, round it up
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }
}


torch::Tensor cu_batch_norm(
    const torch::Tensor input,
    const float gamma,
    const float beta
){
    int batch_size = input.size(0);
    torch::Tensor output = torch::empty({batch_size}, 
  						torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    cudaError_t err;
    // compute mean
    float mean, array_sum;
    float *d_array_sum, *d_mean, *d_partial_sum, *d_var;
    cudaMalloc((void **)&d_partial_sum, MAX_GRID_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array_sum, sizeof(float));
    cudaMalloc((void **)&d_var, sizeof(float));
    cudaMalloc((void **)&d_mean, sizeof(float));

    int blocksInGrid = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assert(blocksInGrid < MAX_GRID_SIZE);
    int roundedBlocksInGrid = round_up_to_power_of_two(blocksInGrid);

    kernel_clear_partialArray<<<1,roundedBlocksInGrid>>>(d_partial_sum);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("clear err: %s\n", cudaGetErrorString(err));
    }

    kernel_parallel_sum<<<blocksInGrid, BLOCK_SIZE>>>(
        batch_size,
        input.data<float>(), 
        d_partial_sum);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("parallel sum err: %s\n", cudaGetErrorString(err));
    }

    kernel_sum_array<<<1, roundedBlocksInGrid>>>(d_partial_sum, d_array_sum);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("partial sum err: %s\n", cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in mean computation: %s\n", cudaGetErrorString(err));
    }
    CP_D2H(&array_sum, d_array_sum, sizeof(float));
    
    mean = array_sum / (float)batch_size;
    printf("cuda_mean: %f\n", mean);
    CP_H2D(d_mean, &mean, sizeof(float));
    // compute variance
    float var_sum, var;
    kernel_clear_partialArray<<<1,roundedBlocksInGrid>>>(d_partial_sum);
    kernel_parallel_variance<<<blocksInGrid, BLOCK_SIZE>>>(
        batch_size, 
        d_mean,
        input.data<float>(),
        d_partial_sum);
    kernel_sum_array<<<1, roundedBlocksInGrid>>>(d_partial_sum, d_array_sum);
    CP_D2H(&var_sum, d_array_sum, sizeof(float));
    var = var_sum / (float)batch_size;
    printf("cuda_var: %f\n", var);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in variance computation: %s\n", cudaGetErrorString(err));
    }

    // normalize
    CP_H2D(d_var, &var, sizeof(float));

    kernel_normalize<<<blocksInGrid, BLOCK_SIZE>>>(
        batch_size, 
        d_mean,
        d_var,
        input.data<float>(),
        output.data<float>(),
        gamma,
        beta);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in norm computaiton: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_partial_sum);
    cudaFree(d_mean);
    cudaFree(d_array_sum);
    cudaFree(d_var);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f_cu_my_BN", &cu_batch_norm, "cuda function my batch norm cpp");
}