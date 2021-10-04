from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(name='depth_rasterization',
      ext_modules=[CUDAExtension('depth_rasterization', [
                                 'depth_rasterization_cuda.cpp', 'depth_rasterization_cuda_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})
