import torch
import torch.nn
import time

import sumMatrix

device = torch.device('cuda')

mat_a = torch.randn(1024).to(device)
mat_b = torch.randn(1024).to(device)
T0 = time.time()
mats_out = sumMatrix.cuda_test(mat_a,mat_b)
T1 = time.time()
torch_mat_out = mat_a + mat_b
T2 = time.time()

print("mat_a: {}".format(mat_a))
print("mat_b: {}".format(mat_b))
print("torch mat_output: {}".format(torch_mat_out))
print("cuda mat_output: {}".format(mats_out[0]))
if torch_mat_out.equal(mats_out[0]):
	print('cuda result correct.')
else:
	print('cuda result wrong.')
_sum_threads = 32
_sum_blocks = int((mat_a.size(0) + _sum_threads - 1) / _sum_threads)
print("grid_dim: {}, block_dim: {}".format(_sum_blocks,_sum_threads))
print("torch time: {}".format(T2-T1))
print("cuda time: {}".format(T1-T0))
print("boost: {}".format(((T2-T1)-(T1-T0))/(T2-T1)))