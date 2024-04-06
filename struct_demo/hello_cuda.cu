#include <ATen/ATen.h>
#include <torch/extension.h>

__global__ void hello_cuda_kernel(float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] + 1;
  }
}

at::Tensor hello_cuda(at::Tensor input) {
  auto output = at::empty_like(input);
  int size = input.numel();
  hello_cuda_kernel<<<(size + 255) / 256, 256>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), size);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hello_cuda", &hello_cuda, "Add 1 to each element of the input tensor (CUDA)");
}