#include <torch/extension.h>

torch::Tensor hello_cuda(torch::Tensor input);

torch::Tensor hello(torch::Tensor input) {
  if (input.is_cuda()) {
    return hello_cuda(input);
  }
  return input + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hello", &hello, "Add 1 to each element of the input tensor");
}
