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
  const int input_size = batch * channels * height * width;
  const int output_size = batch * channels * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "BNLauncher", ([&] {
        const scalar_t *src_ = src.data<scalar_t>();
        scalar_t *dst_ = dst.data<scalar_t>();

        kernel_BN<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,
            0, at::cuda::getCurrentCUDAStream()>>>(
               input_size, channels, height, width, src_, dst_
            );
      }));
  THCudaCheck(cudaGetLastError());
  return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myBatchNorm", &f_batch_norm, "my batch norm");
}