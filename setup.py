import os
from os import path
from setuptools import setup
import glob

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

SOURCE_ROOT_DIR = 'csrc'
current_dir = path.dirname(path.abspath(__file__))
sources = glob.glob(path.join(current_dir, SOURCE_ROOT_DIR, '*'))
sources = [x.replace(current_dir + '/', '') for x in sources if x.endswith('.cu') or x.endswith('.cpp')]

setup(
    name='nlos_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='drs.cuda_ops.backend',
            sources=sources,
            extra_compile_args={
                'nvcc': ['-O3'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
