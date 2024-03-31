#include <torch/extension.h>

// CUDA function
at::Tensor BNLauncher(const at::Tensor& src,
                      const int batch);

// 宏定义
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++函数包装
at::Tensor f_batch_norm(const at::Tensor input) {
  CHECK_INPUT(input);
  at::DeviceGuard guard(input.device());	
  int batch = input.size(0);
  return BNLauncher(input, batch);
}

at::Tensor BNLauncher(const at::Tensor& src,
                      const int batch) {

  at::Tensor dst = at::empty({batch, 2 * channels, height, width},    // 开辟一段存储空间
                             src.options());
  
  N = scr.size[0];
  numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float *d_output;
  cudaMalloc((void**)&d_output, numBlocks * sizeof(float));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "BNLauncher", ([&] {
        const scalar_t *src_ = src.data<scalar_t>();
        scalar_t *dst_ = dst.data<scalar_t>();

        kernel_BN<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,
            0, at::cuda::getCurrentCUDAStream()>>>(
               input_size, channels, height, width, src_, dst_
            );
      }));
  float mean = sumArray(min)
  THCudaCheck(cudaGetLastError());
  cudaFree(d_output);
  return dst;
}

void sumArray(const float *input, float *output) {

    kernel_parallel_summing<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myBatchNorm", &f_batch_norm, "my batch norm");
}