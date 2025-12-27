#!/usr/bin/env python3
"""
RoDe SPMM 编译脚本

用法:
    python compile.py [--clean] [--verbose]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_cuda_arch():
    """获取 CUDA 架构"""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    except ImportError:
        pass
    return "-gencode=arch=compute_75,code=sm_75"


def compile_spmm(clean=False, verbose=False):
    """编译 RoDe SPMM"""

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    # 源文件
    sources = [
        "RoDeSpmm.cu",
        "basic_utils.h",
        "common_utils.h",
        "RoDeSpmm.h",
    ]

    # 检查源文件是否存在
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"错误: 找不到源文件 {src}")
            return False

    # 清理旧的可执行文件
    if clean:
        exe_file = script_dir / "test_spmm"
        if exe_file.exists():
            print(f"清理旧的可执行文件: {exe_file}")
            exe_file.unlink()

    # 编译命令
    cuda_arch = get_cuda_arch()

    cmd = [
        "nvcc",
        "main.cu",
        "RoDeSpmm.cu",
        "-o",
        "test_spmm",
        "-O3",
        "-std=c++17",
        "-I",
        ".",  # 当前目录作为头文件路径
        cuda_arch,
        "-Xcompiler",
        "-O3",
        "-Xcompiler",
        "-std=c++17",
    ]

    print("编译 RoDe SPMM...")
    print(f"命令: {' '.join(cmd)}")
    print()

    try:
        if verbose:
            result = subprocess.run(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ 编译成功!")
            exe_path = script_dir / "test_spmm"
            print(f"可执行文件: {exe_path}")
            return True
        else:
            print("✗ 编译失败!")
            if not verbose:
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            return False

    except FileNotFoundError:
        print("✗ 错误: 找不到 nvcc 编译器")
        print("  请确保 CUDA 已正确安装并添加到 PATH")
        return False
    except Exception as e:
        print(f"✗ 编译错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="编译 RoDe SPMM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python compile.py              # 编译
    python compile.py --clean      # 清理后编译
    python compile.py --verbose    # 详细输出
        """,
    )

    parser.add_argument("--clean", action="store_true", help="清理旧的可执行文件后编译")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    success = compile_spmm(clean=args.clean, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
