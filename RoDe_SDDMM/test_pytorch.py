#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoDe SDDMM PyTorch 接口测试脚本

测试 RoDe SDDMM 与 torch.sparse 操作的结果对比

用法:
    python test_pytorch.py                    # 运行所有测试
    python test_pytorch.py --compile          # 先编译再测试
    python test_pytorch.py --single 512 1024 64  # 单项测试
    python test_pytorch.py --category small   # 按类别测试
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# 尝试导入 tabulate
try:
    from tabulate import tabulate
except ImportError:
    print("警告: 缺少 tabulate 库，将使用简单格式输出")
    print("安装: pip install tabulate")
    tabulate = None


# ============================================================================
# 颜色输出
# ============================================================================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def color_text(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def print_header(msg: str):
    print()
    print(color_text("=" * 70, Colors.CYAN))
    print(color_text(f"  {msg}", Colors.CYAN + Colors.BOLD))
    print(color_text("=" * 70, Colors.CYAN))
    print()


# ============================================================================
# 测试配置
# ============================================================================
@dataclass
class TestConfig:
    m: int
    n: int
    nnz_per_row: int
    desc: str
    category: str = "general"


TEST_CONFIGS: List[TestConfig] = [
    # 小规模测试
    TestConfig(64, 64, 8, "小规模 (64x64)", "small"),
    TestConfig(128, 128, 16, "小规模 (128x128)", "small"),
    TestConfig(256, 256, 32, "小规模 (256x256)", "small"),
    # Irregular 测试
    TestConfig(63, 67, 10, "Irregular (63x67)", "irregular"),
    TestConfig(127, 131, 20, "Irregular (127x131)", "irregular"),
    TestConfig(511, 517, 33, "Irregular (511x517)", "irregular"),
    # 中等规模测试
    TestConfig(512, 512, 32, "中等规模 (512x512)", "medium"),
    TestConfig(1024, 1024, 64, "中等规模 (1024x1024)", "medium"),
    TestConfig(2048, 2048, 128, "中等规模 (2048x2048)", "medium"),
    # 非方阵测试
    TestConfig(2048, 256, 32, "高矩阵 (2048x256)", "rectangular"),
    TestConfig(256, 2048, 32, "宽矩阵 (256x2048)", "rectangular"),
    TestConfig(1999, 257, 45, "高矩阵 irregular (1999x257)", "rectangular"),
    # 稀疏程度测试
    TestConfig(1024, 1024, 4, "极稀疏 (4 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 64, "中等密度 (64 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 256, "较密集 (256 nnz/row)", "sparsity"),
    # 素数维度测试
    TestConfig(127, 131, 23, "素数维度 (127x131)", "prime"),
    TestConfig(509, 521, 67, "素数维度 (509x521)", "prime"),
    # 大规模测试
    TestConfig(4093, 4099, 128, "大规模 irregular (4093x4099)", "large"),
]

CATEGORY_DESCRIPTIONS = {
    "small": "小规模测试",
    "irregular": "Irregular 维度测试",
    "medium": "中等规模测试",
    "rectangular": "非方阵测试",
    "sparsity": "稀疏程度测试",
    "prime": "素数维度测试",
    "large": "大规模测试",
}


# ============================================================================
# 测试结果
# ============================================================================
@dataclass
class TestResult:
    config: TestConfig
    passed: bool
    mae: float = 0.0
    mean_rel: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    rode_time_ms: float = 0.0
    torch_time_ms: float = 0.0
    error_msg: str = ""


# ============================================================================
# 编译扩展
# ============================================================================
def compile_extension(verbose: bool = False) -> bool:
    """编译 PyTorch CUDA 扩展"""
    print_header("编译 RoDe SDDMM PyTorch 扩展")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]

    print(f"命令: {' '.join(cmd)}")
    print()

    try:
        if verbose:
            result = subprocess.run(cmd, timeout=300)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(color_text("✓ 编译成功!", Colors.GREEN))
            return True
        else:
            print(color_text("✗ 编译失败!", Colors.RED))
            if not verbose and hasattr(result, "stderr"):
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(color_text("✗ 编译超时!", Colors.RED))
        return False
    except Exception as e:
        print(color_text(f"✗ 编译错误: {e}", Colors.RED))
        return False


# ============================================================================
# 运行单个测试
# ============================================================================
def run_single_test(m: int, n: int, nnz_per_row: int, k: int = 128) -> TestResult:
    """运行单个测试"""
    from rode_sddmm import RoDeCSR, compare_results, rode_sddmm, torch_sddmm_reference

    config = TestConfig(m, n, nnz_per_row, f"m={m}, n={n}, nnz/row={nnz_per_row}")

    # 创建 RoDe CSR 矩阵
    rode_csr = RoDeCSR.from_random(m, n, nnz_per_row, k=k, seed=42)

    # 创建稠密矩阵
    torch.manual_seed(123)
    lhs = torch.randn(m, k, device="cuda", dtype=torch.float32)
    torch.manual_seed(456)
    rhs = torch.randn(n, k, device="cuda", dtype=torch.float32)

    # 预热
    _ = rode_sddmm(rode_csr, lhs, rhs)
    torch.cuda.synchronize()

    # RoDe SDDMM 计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        rode_output = rode_sddmm(rode_csr, lhs, rhs)
    torch.cuda.synchronize()
    rode_time = (time.perf_counter() - start) / 10 * 1000  # ms

    # PyTorch sparse 参考实现
    torch_sparse_csr = rode_csr.to_torch_sparse_csr()

    # 预热
    _ = torch_sddmm_reference(torch_sparse_csr, lhs, rhs)
    torch.cuda.synchronize()

    # Torch sparse 计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        torch_output = torch_sddmm_reference(torch_sparse_csr, lhs, rhs)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / 10 * 1000  # ms

    # 比较结果
    comparison = compare_results(rode_output, torch_output, rode_csr)

    return TestResult(
        config=config,
        passed=comparison["passed"],
        mae=comparison["mae"],
        mean_rel=comparison["mean_rel"],
        max_abs=comparison["max_abs"],
        max_rel=comparison["max_rel"],
        rode_time_ms=rode_time,
        torch_time_ms=torch_time,
    )


# ============================================================================
# 格式化输出
# ============================================================================
def format_scientific(value: float) -> str:
    return f"{value:.2e}"


def format_result(passed: bool) -> str:
    if passed:
        return color_text("PASS", Colors.GREEN)
    else:
        return color_text("FAIL", Colors.RED)


def format_speedup(rode_time: float, torch_time: float) -> str:
    if rode_time > 0:
        speedup = torch_time / rode_time
        if speedup >= 1:
            return color_text(f"{speedup:.2f}x", Colors.GREEN)
        else:
            return color_text(f"{speedup:.2f}x", Colors.YELLOW)
    return "N/A"


# ============================================================================
# 运行测试并显示表格
# ============================================================================
def run_tests_with_table(configs: List[TestConfig], title: str = "测试结果", k: int = 128) -> Tuple[int, int]:
    """运行测试并显示表格"""
    print_header(title)

    results: List[TestResult] = []
    passed_count = 0
    failed_count = 0

    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        print(f"\r运行测试 [{i}/{total}]: {cfg.desc}...", end="", flush=True)

        result = run_single_test(cfg.m, cfg.n, cfg.nnz_per_row, k)
        result.config = cfg
        results.append(result)

        if result.passed:
            passed_count += 1
        else:
            failed_count += 1

    print("\r" + " " * 80 + "\r", end="")

    # 构建表格
    headers = [
        "#",
        "m",
        "n",
        "nnz/row",
        "MAE",
        "MaxAbs",
        "MaxRel",
        "RoDe(ms)",
        "Torch(ms)",
        "Speedup",
        "Result",
    ]

    table_data = []
    for i, res in enumerate(results, 1):
        if res.error_msg:
            row = [
                i,
                res.config.m,
                res.config.n,
                res.config.nnz_per_row,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                format_result(False) + f" ({res.error_msg[:20]}...)",
            ]
        else:
            row = [
                i,
                res.config.m,
                res.config.n,
                res.config.nnz_per_row,
                format_scientific(res.mae),
                format_scientific(res.max_abs),
                format_scientific(res.max_rel),
                f"{res.rode_time_ms:.3f}",
                f"{res.torch_time_ms:.3f}",
                format_speedup(res.rode_time_ms, res.torch_time_ms),
                format_result(res.passed),
            ]
        table_data.append(row)

    # 打印表格
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    else:
        # 简单格式输出
        print(" | ".join(headers))
        print("-" * 100)
        for row in table_data:
            print(" | ".join(str(x) for x in row))

    print()
    print(
        color_text(f"通过: {passed_count}/{total}", Colors.GREEN if passed_count == total else Colors.YELLOW)
    )
    print(color_text(f"失败: {failed_count}/{total}", Colors.RED if failed_count > 0 else Colors.GREEN))

    return passed_count, failed_count


# ============================================================================
# 运行所有测试
# ============================================================================
def run_all_tests(k: int = 128) -> Tuple[int, int]:
    return run_tests_with_table(TEST_CONFIGS, f"RoDe SDDMM vs torch.sparse 测试 (k={k})", k)


# ============================================================================
# 运行特定类别测试
# ============================================================================
def run_category_tests(category: str, k: int = 128) -> Tuple[int, int]:
    configs = [c for c in TEST_CONFIGS if c.category == category]
    if not configs:
        print(color_text(f"未找到类别: {category}", Colors.RED))
        print("可用类别:", ", ".join(CATEGORY_DESCRIPTIONS.keys()))
        return 0, 0

    desc = CATEGORY_DESCRIPTIONS.get(category, category)
    return run_tests_with_table(configs, f"{desc} (k={k})", k)


# ============================================================================
# 单项详细测试
# ============================================================================
def run_detailed_test(m: int, n: int, nnz_per_row: int, k: int = 128):
    """运行单项测试并显示详细结果"""
    print_header(f"详细测试: m={m}, n={n}, nnz_per_row={nnz_per_row}, k={k}")

    from rode_sddmm import RoDeCSR, compare_results, rode_sddmm, torch_sddmm_reference

    # 创建矩阵
    print("创建 RoDe CSR 矩阵...")
    rode_csr = RoDeCSR.from_random(m, n, nnz_per_row, k=k, seed=42)
    print(f"  {rode_csr}")
    print()

    # 创建稠密矩阵
    print("创建稠密矩阵...")
    torch.manual_seed(123)
    lhs = torch.randn(m, k, device="cuda", dtype=torch.float32)
    torch.manual_seed(456)
    rhs = torch.randn(n, k, device="cuda", dtype=torch.float32)
    print(f"  lhs: {lhs.shape}")
    print(f"  rhs: {rhs.shape}")
    print()

    # RoDe SDDMM
    print("执行 RoDe SDDMM...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    rode_output = rode_sddmm(rode_csr, lhs, rhs)
    torch.cuda.synchronize()
    rode_time = (time.perf_counter() - start) * 1000
    print(f"  输出形状: {rode_output.shape}")
    print(f"  耗时: {rode_time:.3f} ms")
    print()

    # PyTorch sparse 参考
    print("执行 torch.sparse 参考实现...")
    torch_sparse_csr = rode_csr.to_torch_sparse_csr()
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch_output = torch_sddmm_reference(torch_sparse_csr, lhs, rhs)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) * 1000
    print(f"  输出形状: {torch_output.shape}")
    print(f"  耗时: {torch_time:.3f} ms")
    print()

    # 比较结果
    print("比较结果...")
    comparison = compare_results(rode_output, torch_output, rode_csr)

    # 显示详细表格
    detail_data = [
        ["参数 m", m],
        ["参数 n", n],
        ["参数 nnz_per_row", nnz_per_row],
        ["参数 k", k],
        ["实际 nnz", rode_csr.nnz],
        ["Block 部分数量 (m1)", rode_csr.m1],
        ["Residue 部分数量 (m2)", rode_csr.m2],
        ["MAE (平均绝对误差)", format_scientific(comparison["mae"])],
        ["平均相对误差", format_scientific(comparison["mean_rel"])],
        ["最大绝对误差", format_scientific(comparison["max_abs"])],
        ["最大相对误差", format_scientific(comparison["max_rel"])],
        ["错误元素数", comparison.get("error_count", 0)],
        ["RoDe 耗时", f"{rode_time:.3f} ms"],
        ["Torch 耗时", f"{torch_time:.3f} ms"],
        ["加速比", format_speedup(rode_time, torch_time)],
        ["测试结果", format_result(comparison["passed"])],
    ]

    if tabulate:
        print(tabulate(detail_data, tablefmt="simple"))
    else:
        for row in detail_data:
            print(f"  {row[0]}: {row[1]}")

    print()
    return comparison["passed"]


# ============================================================================
# 列出类别
# ============================================================================
def list_categories():
    print_header("可用测试类别")

    data = []
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        count = len([c for c in TEST_CONFIGS if c.category == cat])
        data.append([cat, desc, count])

    if tabulate:
        print(tabulate(data, headers=["类别ID", "描述", "测试数量"], tablefmt="simple"))
    else:
        print("类别ID\t\t描述\t\t测试数量")
        for row in data:
            print(f"{row[0]}\t\t{row[1]}\t\t{row[2]}")

    print()
    print(f"总测试数: {len(TEST_CONFIGS)}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RoDe SDDMM PyTorch 接口测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python test_pytorch.py                       # 运行所有测试
    python test_pytorch.py --compile             # 先编译再测试
    python test_pytorch.py --compile-only        # 仅编译
    python test_pytorch.py --single 512 1024 64  # 单项详细测试
    python test_pytorch.py --category small      # 按类别测试
    python test_pytorch.py --list-categories     # 列出所有类别
    python test_pytorch.py --k 32                # 使用 k=32 版本
        """,
    )

    parser.add_argument("--compile", action="store_true", help="编译后运行测试")
    parser.add_argument("--compile-only", action="store_true", help="仅编译，不运行测试")
    parser.add_argument("--single", nargs=3, type=int, metavar=("M", "N", "NNZ"), help="运行单项详细测试")
    parser.add_argument("--category", type=str, metavar="CAT", help="运行特定类别测试")
    parser.add_argument("--list-categories", action="store_true", help="列出所有测试类别")
    parser.add_argument("--k", type=int, default=128, choices=[32, 128], help="隐藏维度 k (默认: 128)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    print(
        color_text(
            """
    ╔═══════════════════════════════════════════════════════════════╗
    ║        RoDe SDDMM PyTorch 接口测试                            ║
    ║        对比 RoDe vs torch.sparse                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """,
            Colors.CYAN + Colors.BOLD,
        )
    )

    # 切换目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 列出类别
    if args.list_categories:
        list_categories()
        sys.exit(0)

    # 编译
    if args.compile or args.compile_only:
        if not compile_extension(args.verbose):
            sys.exit(1)
        if args.compile_only:
            sys.exit(0)

    # 检查扩展是否可用
    try:
        import rode_sddmm_cuda

        print(color_text("✓ rode_sddmm_cuda 扩展已加载", Colors.GREEN))
    except ImportError:
        print(color_text("✗ rode_sddmm_cuda 扩展未找到，请先编译:", Colors.RED))
        print("    python setup.py build_ext --inplace")
        print("  或者:")
        print("    python test_pytorch.py --compile")
        sys.exit(1)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print(color_text("✗ CUDA 不可用!", Colors.RED))
        sys.exit(1)

    print(f"CUDA 设备: {torch.cuda.get_device_name()}")
    print()

    # 运行测试
    success = True
    start_time = time.time()

    if args.single:
        m, n, nnz_per_row = args.single
        success = run_detailed_test(m, n, nnz_per_row, args.k)

    elif args.category:
        passed, failed = run_category_tests(args.category, args.k)
        success = failed == 0

    else:
        passed, failed = run_all_tests(args.k)
        success = failed == 0

    # 打印总耗时
    elapsed = time.time() - start_time
    print()
    print(color_text(f"总耗时: {elapsed:.2f} 秒", Colors.CYAN))

    if success:
        print(color_text("\n✓ 测试完成!", Colors.GREEN + Colors.BOLD))
        sys.exit(0)
    else:
        print(color_text("\n✗ 测试失败!", Colors.RED + Colors.BOLD))
        sys.exit(1)


if __name__ == "__main__":
    main()
