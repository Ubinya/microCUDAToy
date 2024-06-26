from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

PACKAGE_NAME = 'myBatchNorm'
DESCRIPTION = 'a batch norm test module based on CUDA.'
VERSION = '0.1.0'
setup(
	name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    license='MIT',
    ext_modules=[
        CUDAExtension(
        	name='my_batch_norm',
        	sources=['BatchNorm_kernel.cu'],
        	extra_compile_args={'cxx':['-std=c++17', '-ffast-math'],'nvcc':['-std=c++17']},
            extra_link_args=[])],
        cmdclass={'build_ext':BuildExtension}
	)