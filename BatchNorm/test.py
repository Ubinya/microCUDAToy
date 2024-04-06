import torch
import torch.nn.functional as F
import torch.nn
import time

import my_batch_norm

def batch_norm_1d(input_tensor, gamma, beta, eps=1e-5):
    # Calculate mean and variance
    mean = torch.mean(input_tensor)
    var = torch.var(input_tensor, unbiased=False)

    # Normalize input_tensor
    normalized_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # Scale and shift using gamma and beta parameters
    output_tensor = gamma * normalized_tensor + beta

    return output_tensor

input_tensor = torch.randn(1024).to('cuda')
# Example usage
gamma = torch.tensor(0.8).to('cuda')  # Scaling parameter
beta = torch.tensor(0.1).to('cuda')   # Shift parameter


T0 = time.time()
torch_tensor = batch_norm_1d(input_tensor, gamma, beta)
T1 = time.time()
cuda_tensor = my_batch_norm.f_cu_my_BN(input_tensor, 0.8, 0.1)
T2 = time.time()

print("input_tensor: {}".format(input_tensor))
print("torch mat_output: {}".format(torch_tensor))
print("cuda mat_output: {}".format(cuda_tensor))
tolerance = 1e-5
print('under 1e-5 differences')
if torch.allclose(torch_tensor, cuda_tensor, atol=tolerance):
	print('cuda result correct.')
else:
	print('cuda result wrong.')
print("torch time: {}".format(T1-T0))
print("cuda time: {}".format(T2-T1))
print("boost: {}".format(((T1-T0)-(T2-T1))/(T1-T0)))



