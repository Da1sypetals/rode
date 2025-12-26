"""
RoDe SDDMM PyTorch 扩展编译脚本

用法:
    python setup.py install        # 安装扩展
    python setup.py build_ext --inplace  # 就地编译
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_arch():
    """获取 CUDA 架构"""
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return "-gencode=arch=compute_75,code=sm_75"


# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 源文件列表
sources = [
    os.path.join(current_dir, "rode_sddmm_cuda.cu"),
    os.path.join(current_dir, "RoDeSddmm.cu"),
]

# 头文件目录
include_dirs = [current_dir]

# NVCC 编译选项
nvcc_flags = [
    "-O3",
    "-std=c++17",
    get_cuda_arch(),
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
]

# CXX 编译选项
cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="rode_sddmm_cuda",
    version="1.0.0",
    description="RoDe SDDMM CUDA Extension for PyTorch",
    author="RoDe Team",
    ext_modules=[
        CUDAExtension(
            name="rode_sddmm_cuda",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
    ],
)
