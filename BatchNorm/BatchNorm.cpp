#include <torch/extension.h>

// CUDA function
at::Tensor NCReLUForwardLauncher(const at::Tensor& src,
                                 const int batch,
                                 const int channels,
                                 const int height,
                                 const int width);

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
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  return NCReLUForwardLauncher(input, batch, channels, height, width);
}

std::vector<torch::Tensor> f_matrix_sum_old(
    const torch::Tensor mat_in) {

    int n = mat_a.size(0);

    const int _sum_threads = 32; // threads per block
    const int _sum_blocks = (n + _sum_threads - 1) / _sum_threads;

    // create an empty tensor to store the output.
    torch::Tensor mat_out = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    kernel_sum_mat <<<_sum_blocks, _sum_threads>>> (
            mat_in.data<float>(),
            n); 

    return {mat_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myBatchNorm", &f_batch_norm, "my batch norm");
}