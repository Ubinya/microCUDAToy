from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='hello',
    ext_modules=[
        CUDAExtension('hello_cuda', [
            'hello_cuda.cpp',
            'hello_cuda.cu',
        ],
        extra_link_args=[]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
