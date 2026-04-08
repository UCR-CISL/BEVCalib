from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_pool_ext',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_ext',
            sources=[
                'src/bev_pool_cpp.cpp',
                'src/bev_pool_cuda.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)