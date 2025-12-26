#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoDe SDDMM 测试脚本 (k=128)

测试各种 m, n 值（包括 irregular 值）和不同稀疏程度的正确性
所有测试参数在本脚本中定义，CUDA 程序只负责执行并返回 JSON 结果

依赖:
    pip install tabulate

用法:
    python run_test.py                     # 编译并运行所有测试
    python run_test.py --compile-only      # 仅编译
    python run_test.py --run-only          # 仅运行（假设已编译）
    python run_test.py --single 1024 2048 64  # 运行单个测试
    python run_test.py --stress 20         # 压力测试 20 次
    python run_test.py --category small    # 运行特定类别的测试
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate


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
    """返回带颜色的文本"""
    return f"{color}{text}{Colors.RESET}"


def print_color(msg: str, color: str = Colors.RESET):
    """彩色打印"""
    print(color_text(msg, color))


def print_header(msg: str):
    """打印标题"""
    print()
    print_color("=" * 70, Colors.CYAN)
    print_color(f"  {msg}", Colors.CYAN + Colors.BOLD)
    print_color("=" * 70, Colors.CYAN)
    print()


# ============================================================================
# 测试配置
# ============================================================================
@dataclass
class TestConfig:
    """测试配置"""

    m: int
    n: int
    nnz_per_row: int
    desc: str
    category: str = "general"


# 所有测试用例定义
TEST_CONFIGS: List[TestConfig] = [
    # 小规模 regular 测试
    TestConfig(64, 64, 8, "小规模 regular (64x64)", "small"),
    TestConfig(128, 128, 16, "小规模 regular (128x128)", "small"),
    TestConfig(256, 256, 32, "小规模 regular (256x256)", "small"),
    # 小规模 irregular 测试
    TestConfig(63, 67, 10, "小规模 irregular (63x67)", "small_irregular"),
    TestConfig(97, 103, 15, "小规模 irregular (97x103)", "small_irregular"),
    TestConfig(127, 131, 20, "小规模 irregular (127x131)", "small_irregular"),
    # 中等规模 regular 测试
    TestConfig(512, 512, 32, "中等规模 regular (512x512)", "medium"),
    TestConfig(1024, 1024, 64, "中等规模 regular (1024x1024)", "medium"),
    TestConfig(2048, 2048, 128, "中等规模 regular (2048x2048)", "medium"),
    # 中等规模 irregular 测试
    TestConfig(511, 517, 33, "中等规模 irregular (511x517)", "medium_irregular"),
    TestConfig(1023, 1031, 65, "中等规模 irregular (1023x1031)", "medium_irregular"),
    TestConfig(2047, 2053, 127, "中等规模 irregular (2047x2053)", "medium_irregular"),
    # 非方阵测试 (m >> n)
    TestConfig(2048, 256, 32, "高矩阵 (2048x256)", "tall"),
    TestConfig(4096, 512, 64, "高矩阵 (4096x512)", "tall"),
    TestConfig(1999, 257, 45, "高矩阵 irregular (1999x257)", "tall"),
    # 非方阵测试 (m << n)
    TestConfig(256, 2048, 32, "宽矩阵 (256x2048)", "wide"),
    TestConfig(512, 4096, 64, "宽矩阵 (512x4096)", "wide"),
    TestConfig(257, 2003, 47, "宽矩阵 irregular (257x2003)", "wide"),
    # 极端 irregular 维度
    TestConfig(1, 1000, 100, "单行矩阵 (1x1000)", "extreme"),
    TestConfig(1000, 1, 1, "单列矩阵 (1000x1)", "extreme"),
    TestConfig(17, 1999, 150, "极端 irregular (17x1999)", "extreme"),
    TestConfig(1999, 17, 5, "极端 irregular (1999x17)", "extreme"),
    TestConfig(31, 4097, 200, "极端 irregular (31x4097)", "extreme"),
    # 不同稀疏程度测试 (固定维度 1024x1024)
    TestConfig(1024, 1024, 1, "极稀疏 (1 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 4, "很稀疏 (4 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 16, "稀疏 (16 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 64, "中等密度 (64 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 256, "较密集 (256 nnz/row)", "sparsity"),
    TestConfig(1024, 1024, 512, "密集 (512 nnz/row)", "sparsity"),
    # 边界测试
    TestConfig(32, 32, 16, "边界测试 (32x32)", "boundary"),
    TestConfig(33, 31, 10, "边界测试 irregular (33x31)", "boundary"),
    TestConfig(128, 32, 8, "边界测试 (128x32)", "boundary"),
    TestConfig(32, 128, 8, "边界测试 (32x128)", "boundary"),
    # 素数维度测试
    TestConfig(127, 131, 23, "素数维度 (127x131)", "prime"),
    TestConfig(251, 257, 47, "素数维度 (251x257)", "prime"),
    TestConfig(509, 521, 67, "素数维度 (509x521)", "prime"),
    TestConfig(1021, 1031, 97, "素数维度 (1021x1031)", "prime"),
    # 大规模测试
    TestConfig(4096, 4096, 128, "大规模 (4096x4096)", "large"),
    TestConfig(4093, 4099, 129, "大规模 irregular (4093x4099)", "large"),
]

# 测试类别描述
CATEGORY_DESCRIPTIONS = {
    "small": "小规模 regular 测试",
    "small_irregular": "小规模 irregular 测试",
    "medium": "中等规模 regular 测试",
    "medium_irregular": "中等规模 irregular 测试",
    "tall": "高矩阵测试 (m >> n)",
    "wide": "宽矩阵测试 (m << n)",
    "extreme": "极端维度测试",
    "sparsity": "稀疏程度测试",
    "boundary": "边界测试",
    "prime": "素数维度测试",
    "large": "大规模测试",
}


# ============================================================================
# 测试结果
# ============================================================================
@dataclass
class TestResult:
    """测试结果"""

    config: TestConfig
    passed: bool
    mae: float = 0.0
    mean_rel: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    errors: int = 0
    nnz: int = 0
    error_msg: str = ""
    timeout: bool = False


# ============================================================================
# GPU 架构检测
# ============================================================================
def detect_gpu_arch() -> str:
    """自动检测 GPU 架构"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            cap = result.stdout.strip().split("\n")[0].replace(".", "")
            return f"sm_{cap}"
    except Exception:
        pass
    return "sm_75"


# ============================================================================
# 编译
# ============================================================================
def compile_test(arch: Optional[str] = None, verbose: bool = False) -> bool:
    """编译测试程序"""
    print_header("编译测试程序")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if arch is None:
        arch = detect_gpu_arch()

    print(f"GPU 架构: {arch}")
    print(f"工作目录: {script_dir}")
    print()

    nvcc_cmd = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "-std=c++17",
        "-o",
        "test_k128",
        "test.cu",
        "RoDeSddmm.cu",
        "-I.",
    ]

    print(f"编译命令: {' '.join(nvcc_cmd)}")
    print()

    try:
        result = subprocess.run(nvcc_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print_color("✓ 编译成功!", Colors.GREEN)
            return True
        else:
            print_color("✗ 编译失败!", Colors.RED)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print_color("✗ 编译超时!", Colors.RED)
        return False
    except FileNotFoundError:
        print_color("✗ 找不到 nvcc 编译器!", Colors.RED)
        return False


# ============================================================================
# 运行单个测试
# ============================================================================
def run_test(m: int, n: int, nnz_per_row: int, timeout: int = 60) -> Dict[str, Any]:
    """运行单个测试并返回 JSON 结果"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(script_dir, "test_k128")

    if not os.path.exists(exe_path):
        return {"passed": False, "error": "测试程序不存在，请先编译"}

    try:
        result = subprocess.run(
            [exe_path, str(m), str(n), str(nnz_per_row)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_dir,
        )

        # 解析 JSON 输出
        output = result.stdout.strip()
        if output:
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return {"passed": False, "error": f"JSON 解析失败: {output}"}
        else:
            return {"passed": False, "error": f"无输出, stderr: {result.stderr}"}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "超时", "timeout": True}
    except Exception as e:
        return {"passed": False, "error": str(e)}


# ============================================================================
# 格式化数值
# ============================================================================
def format_scientific(value: float) -> str:
    """格式化科学计数法"""
    return f"{value:.2e}"


def format_result(passed: bool) -> str:
    """格式化测试结果"""
    if passed:
        return color_text("PASS", Colors.GREEN)
    else:
        return color_text("FAIL", Colors.RED)


# ============================================================================
# 运行测试并显示表格
# ============================================================================
def run_tests_with_table(
    configs: List[TestConfig], title: str = "测试结果", show_description: bool = True
) -> Tuple[int, int]:
    """运行测试并用表格显示结果"""
    print_header(title)

    results: List[TestResult] = []
    passed_count = 0
    failed_count = 0

    # 运行所有测试
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        print(f"\r运行测试 [{i}/{total}]: {cfg.desc}...")

        result_dict = run_test(cfg.m, cfg.n, cfg.nnz_per_row)

        result = TestResult(
            config=cfg,
            passed=result_dict.get("passed", False),
            mae=result_dict.get("mae", 0.0),
            mean_rel=result_dict.get("mean_rel", 0.0),
            max_abs=result_dict.get("max_abs", 0.0),
            max_rel=result_dict.get("max_rel", 0.0),
            errors=result_dict.get("errors", 0),
            nnz=result_dict.get("nnz", 0),
            error_msg=result_dict.get("error", ""),
            timeout=result_dict.get("timeout", False),
        )
        results.append(result)

        if result.passed:
            passed_count += 1
        else:
            failed_count += 1

    # 清除进度信息
    print("\r" + " " * 80 + "\r", end="")

    # 构建表格数据
    headers = ["#", "m", "n", "nnz/row", "MAE", "MeanRel", "MaxAbs", "MaxRel", "Result"]
    if show_description:
        headers.append("Description")

    table_data = []
    for i, res in enumerate(results, 1):
        row = [
            i,
            res.config.m,
            res.config.n,
            res.config.nnz_per_row,
            format_scientific(res.mae) if res.passed or not res.timeout else "N/A",
            format_scientific(res.mean_rel) if res.passed or not res.timeout else "N/A",
            format_scientific(res.max_abs) if res.passed or not res.timeout else "N/A",
            format_scientific(res.max_rel) if res.passed or not res.timeout else "N/A",
            format_result(res.passed) + (f" ({res.error_msg})" if res.error_msg and not res.passed else ""),
        ]
        if show_description:
            row.append(res.config.desc)
        table_data.append(row)

    # 打印表格
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print()

    # 打印统计
    print_color(f"通过: {passed_count}/{total}", Colors.GREEN if passed_count == total else Colors.YELLOW)
    print_color(f"失败: {failed_count}/{total}", Colors.RED if failed_count > 0 else Colors.GREEN)

    return passed_count, failed_count


# ============================================================================
# 运行所有测试
# ============================================================================
def run_all_tests() -> Tuple[int, int]:
    """运行所有预设测试"""
    return run_tests_with_table(TEST_CONFIGS, "RoDe SDDMM 全面测试 (k=128)")


# ============================================================================
# 运行特定类别测试
# ============================================================================
def run_category_tests(category: str) -> Tuple[int, int]:
    """运行特定类别的测试"""
    configs = [c for c in TEST_CONFIGS if c.category == category]
    if not configs:
        print_color(f"未找到类别: {category}", Colors.RED)
        print("可用类别:")
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            print(f"  {cat}: {desc}")
        return 0, 0

    desc = CATEGORY_DESCRIPTIONS.get(category, category)
    return run_tests_with_table(configs, f"{desc} (共 {len(configs)} 项)")


# ============================================================================
# 运行单个测试
# ============================================================================
def run_single_test(m: int, n: int, nnz_per_row: int) -> bool:
    """运行单个测试并详细显示结果"""
    print_header(f"单项测试: m={m}, n={n}, nnz_per_row={nnz_per_row}")

    result = run_test(m, n, nnz_per_row)

    # 构建详细表格
    if result.get("passed") is not None:
        detail_data = [
            ["参数 m", m],
            ["参数 n", n],
            ["参数 nnz_per_row", nnz_per_row],
            ["参数 k", 128],
            ["实际 nnz", result.get("nnz", "N/A")],
            ["MAE (平均绝对误差)", format_scientific(result.get("mae", 0))],
            ["平均相对误差", format_scientific(result.get("mean_rel", 0))],
            ["最大绝对误差", format_scientific(result.get("max_abs", 0))],
            ["最大相对误差", format_scientific(result.get("max_rel", 0))],
            ["错误元素数", result.get("errors", 0)],
            ["测试结果", format_result(result.get("passed", False))],
        ]
        print(tabulate(detail_data, tablefmt="simple"))
    else:
        print_color(f"错误: {result.get('error', '未知错误')}", Colors.RED)

    print()
    return result.get("passed", False)


# ============================================================================
# 压力测试
# ============================================================================
def run_stress_test(iterations: int = 10, seed: int = 42) -> Tuple[int, int]:
    """压力测试：随机参数多次运行"""
    random.seed(seed)

    # 生成随机测试配置
    configs = []
    for i in range(iterations):
        m = random.randint(10, 2000)
        n = random.randint(10, 2000)
        nnz_per_row = random.randint(1, min(n, 200))
        configs.append(TestConfig(m, n, nnz_per_row, f"随机测试 #{i + 1}", "stress"))

    return run_tests_with_table(configs, f"压力测试 ({iterations} 次随机参数)", show_description=False)


# ============================================================================
# 列出所有类别
# ============================================================================
def list_categories():
    """列出所有测试类别"""
    print_header("可用测试类别")

    data = []
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        count = len([c for c in TEST_CONFIGS if c.category == cat])
        data.append([cat, desc, count])

    print(tabulate(data, headers=["类别ID", "描述", "测试数量"], tablefmt="simple"))
    print()
    print(f"总测试数: {len(TEST_CONFIGS)}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RoDe SDDMM 测试脚本 (k=128)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_test.py                        # 编译并运行所有测试
  python run_test.py --compile-only         # 仅编译
  python run_test.py --run-only             # 仅运行
  python run_test.py --single 1024 2048 64  # 单项测试
  python run_test.py --category small       # 运行特定类别
  python run_test.py --stress 20            # 压力测试 20 次
  python run_test.py --list-categories      # 列出所有类别
  python run_test.py --arch sm_80           # 指定 GPU 架构
        """,
    )

    parser.add_argument("--compile-only", action="store_true", help="仅编译，不运行测试")
    parser.add_argument("--run-only", action="store_true", help="仅运行测试（需要已编译）")
    parser.add_argument("--single", nargs=3, type=int, metavar=("M", "N", "NNZ"), help="运行单项测试")
    parser.add_argument("--category", type=str, metavar="CAT", help="运行特定类别的测试")
    parser.add_argument("--stress", type=int, metavar="N", help="运行 N 次压力测试")
    parser.add_argument("--list-categories", action="store_true", help="列出所有测试类别")
    parser.add_argument("--arch", type=str, default=None, help="指定 GPU 架构 (如 sm_75, sm_80)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    print_color("=" * 30, Colors.CYAN + Colors.BOLD)
    print_color("RoDe SDDMM 测试脚本 (k=128)", Colors.CYAN + Colors.BOLD)
    print_color("=" * 30, Colors.CYAN + Colors.BOLD)

    # 列出类别
    if args.list_categories:
        list_categories()
        sys.exit(0)

    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 处理编译
    if not args.run_only:
        if not compile_test(args.arch, args.verbose):
            sys.exit(1)

    if args.compile_only:
        sys.exit(0)

    # 处理测试
    success = True
    start_time = time.time()

    if args.single:
        m, n, nnz_per_row = args.single
        success = run_single_test(m, n, nnz_per_row)

    elif args.category:
        passed, failed = run_category_tests(args.category)
        success = failed == 0

    elif args.stress:
        passed, failed = run_stress_test(args.stress)
        success = failed == 0

    else:
        passed, failed = run_all_tests()
        success = failed == 0

    # 打印总耗时
    elapsed = time.time() - start_time
    print()
    print_color(f"总耗时: {elapsed:.2f} 秒", Colors.CYAN)

    if success:
        print_color("\n✓ 测试完成！", Colors.GREEN + Colors.BOLD)
        sys.exit(0)
    else:
        print_color("\n✗ 测试失败！", Colors.RED + Colors.BOLD)
        sys.exit(1)


if __name__ == "__main__":
    main()
