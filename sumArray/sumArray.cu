#include <stdio.h>
#include <cassert>

#define N 10000
#define BLOCK_SIZE 256
#define MAX_GRID_SIZE 512

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
    if (index < N) {
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
        __syncthreads(); // Synchronize threads within the block
    }

    // Store the partial sum for this block to global memory
    if (tid == 0) {
        dst_data[blockIdx.x] = partialArr[0];
    }
}

__global__ void kernel_sum_array(
    float *input, 
    float *output) {

    __shared__ float partialArr[BLOCK_SIZE];
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


int main(void) {
    float *a, *sum;  // Host copies of a, b, c
    float *d_a, *d_sum, *d_partial_sum;  // Device copies of a, b, c
    int size = N * sizeof(float);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_partial_sum, MAX_GRID_SIZE * sizeof(float));
    cudaMalloc((void **)&d_sum, sizeof(float));

    // Allocate space for host copies of a, b, c and setup input values
    a = (float *)malloc(size);
    sum = (float *)malloc(sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
    }
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    int blocksInGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assert(blocksInGrid < MAX_GRID_SIZE);
    int roundedBlocksInGrid = round_up_to_power_of_two(blocksInGrid);
    // clear partial array
    kernel_clear_partialArray<<<1,roundedBlocksInGrid>>>(d_partial_sum);
    // sum partial array
    kernel_parallel_sum<<<blocksInGrid, BLOCK_SIZE>>>(N, d_a, d_partial_sum);
    printf("grid = %d\n", blocksInGrid);
    // final sum
    kernel_sum_array<<<1, roundedBlocksInGrid>>>(d_partial_sum, d_sum);
    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %f\n", *sum);

    free(a);
    free(sum);
    cudaFree(d_a);
    cudaFree(d_sum);
    cudaFree(d_partial_sum);

    return 0;
}
