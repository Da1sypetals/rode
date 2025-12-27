#!/usr/bin/env python3
"""
RoDe SPMM 测试脚本
测试 C = A × B，其中 A 是 CSR 格式的稀疏矩阵，B 是稠密矩阵
"""

import sys
import subprocess
import time
from pathlib import Path


class Colors:
    """终端颜色"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


class SPMMTest:
    """SPMM 测试类"""

    def __init__(self, executable_path):
        self.executable = Path(executable_path)
        if not self.executable.exists():
            raise FileNotFoundError(f"可执行文件不存在: {executable_path}")
        self.passed = 0
        self.failed = 0

    def run_test(self, m, n, k, nnz_per_row=32, k_version=None):
        """运行单个测试"""
        cmd = [
            str(self.executable),
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--nnz_per_row",
            str(nnz_per_row),
        ]

        if k_version is not None:
            cmd.extend(["--k_version", str(k_version)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # 检查是否通过
            if "✓ 测试通过!" in result.stdout:
                self.passed += 1
                return True, result.stdout
            else:
                self.failed += 1
                return False, result.stdout

        except subprocess.TimeoutExpired:
            self.failed += 1
            return False, "测试超时"
        except Exception as e:
            self.failed += 1
            return False, str(e)

    def print_summary(self):
        """打印测试总结"""
        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}测试总结:{Colors.END}")
        print(f"  总计: {total}")
        print(f"  {Colors.GREEN}通过: {self.passed}{Colors.END}")
        if self.failed > 0:
            print(f"  {Colors.RED}失败: {self.failed}{Colors.END}")
        return self.failed == 0


def print_banner():
    """打印标题"""
    print(f"""
{Colors.BLUE}{"=" * 70}{Colors.END}
{Colors.BOLD}           RoDe SPMM 参数测试{Colors.END}
{Colors.BLUE}{"=" * 70}{Colors.END}
    """)


def print_parameters_limitations():
    """打印参数限制"""
    print(f"""
{Colors.YELLOW}参数限制（基于测试验证）:{Colors.END}

{Colors.BOLD}m 参数（稀疏矩阵行数）:{Colors.END}
  限制: 无限制 ✓
  说明: 支持任意正整数
  典型值: 64, 128, 256, 512, 1024, 4096, 16384, 65536

{Colors.BOLD}k 参数（稀疏矩阵列数 / 稠密矩阵行数）:{Colors.END}
  限制: 无限制 ✓
  说明: 支持任意正整数
  典型值: 64, 128, 256, 512, 1024

{Colors.BOLD}n 参数（稠密矩阵列数 / 输出维度）:{Colors.END}
  限制: 必须是 64 的倍数 ⚠
  说明: n % 64 == 0
  有效值: 64, 128, 192, 256, 320, 384, 448, 512, ...
  无效值: 1-63, 65-127, 129-191, 193-255, ...

{Colors.BOLD}k_version 参数（内部优化参数）:{Colors.END}
  说明: 自动选择，无需手动指定
    - k <= 32: 使用 k_version=32
    - k > 32: 使用 k_version=128
  影响: 仅影响性能，不影响正确性

{Colors.BLUE}{"-" * 70}{Colors.END}
    """)


def run_boundary_tests(tester):
    """运行边界测试"""
    print(f"\n{Colors.BOLD}1. 边界测试{Colors.END}\n")

    # 最小值测试
    print("  最小值测试:")
    test_cases = [
        ("m=1, n=64, k=1", 1, 64, 1, 1),
        ("m=1, n=64, k=64", 1, 64, 64, 1),
        ("m=64, n=64, k=1", 64, 64, 1, 1),
    ]

    for name, m, n, k, nnz in test_cases:
        success, output = tester.run_test(m, n, k, nnz)
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if success else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"    {name:20s} {status}")

    # 素数测试
    print("\n  素数测试（验证无限制参数）:")
    primes = [7, 13, 31, 43, 61, 73, 97, 101, 151, 199, 401, 1009, 4093]
    for p in primes:
        success, _ = tester.run_test(p, 64, p, 32)
        status = f"{Colors.GREEN}✓{Colors.END}" if success else f"{Colors.RED}✗{Colors.END}"
        print(f"    m=k={p:5d} {status}", end="")
        if p % 10 == 1:
            print()
    if primes[-1] % 10 != 1:
        print()

    print()


def run_gnn_tests(tester):
    """运行GNN典型用例测试"""
    print(f"{Colors.BOLD}2. GNN 典型用例测试{Colors.END}\n")

    gnn_cases = [
        ("小型图 (64节点, 64特征)", 64, 64, 64, 16),
        ("小型图 (64节点, 128特征)", 64, 128, 64, 16),
        ("中型图 (512节点, 64特征)", 512, 64, 512, 32),
        ("中型图 (512节点, 128特征)", 512, 128, 512, 32),
        ("中型图 (1024节点, 64特征)", 1024, 64, 1024, 32),
        ("中型图 (1024节点, 128特征)", 1024, 128, 1024, 32),
        ("中型图 (1024节点, 256特征)", 1024, 256, 1024, 64),
        ("大型图 (4096节点, 64特征)", 4096, 64, 4096, 64),
        ("大型图 (4096节点, 128特征)", 4096, 128, 4096, 64),
        ("大型图 (4096节点, 256特征)", 4096, 256, 4096, 128),
        ("超大型图 (16384节点, 64特征)", 16384, 64, 16384, 64),
        ("超大型图 (16384节点, 128特征)", 16384, 128, 16384, 128),
    ]

    for name, m, n, k, nnz in gnn_cases:
        success, output = tester.run_test(m, n, k, nnz)
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if success else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"  {name:35s} {status}")

    print()


def run_n_multiplier_tests(tester):
    """测试n的64倍数限制"""
    print(f"{Colors.BOLD}3. n 的 64 倍数测试{Colors.END}\n")

    n_multipliers = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]

    print("  测试 m=64, k=64, n=64*multiplier:")
    for mult in n_multipliers:
        n = 64 * mult
        success, output = tester.run_test(64, n, 64, 16)
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if success else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"    n = {n:5d} (64×{mult:2d}) {status}")

    print()


def run_k_version_tests(tester):
    """测试k_version的影响"""
    print(f"{Colors.BOLD}4. k_version 自动选择测试{Colors.END}\n")

    print("  k <= 32 (应使用 k_version=32):")
    for k in [1, 8, 16, 24, 32]:
        success, _ = tester.run_test(64, 64, k, 16, k_version=32)
        status = f"{Colors.GREEN}✓{Colors.END}" if success else f"{Colors.RED}✗{Colors.END}"
        print(f"    k={k:2d} {status}", end="")
    print()

    print("\n  k > 32 (应使用 k_version=128):")
    for k in [33, 64, 96, 128, 256]:
        success, _ = tester.run_test(64, 64, k, 16, k_version=128)
        status = f"{Colors.GREEN}✓{Colors.END}" if success else f"{Colors.RED}✗{Colors.END}"
        print(f"    k={k:3d} {status}", end="")
    print("\n")


def run_large_scale_tests(tester):
    """大规模测试"""
    print(f"{Colors.BOLD}5. 大规模测试{Colors.END}\n")

    large_cases = [
        ("大规模 1", 10000, 64, 10000, 100),
        ("大规模 2", 20000, 128, 20000, 100),
        ("大规模 3", 32768, 64, 32768, 128),
        ("大规模 4", 65536, 64, 65536, 128),
    ]

    for name, m, n, k, nnz in large_cases:
        print(f"  {name}: ", end="", flush=True)
        success, output = tester.run_test(m, n, k, nnz)
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if success else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"{status}")
        if success:
            # 提取性能信息
            for line in output.split("\n"):
                if "GPU 耗时" in line:
                    print(f"    {line.strip()}")
                    break

    print()


def main():
    """主函数"""
    print_banner()
    print_parameters_limitations()

    # 检查可执行文件
    script_dir = Path(__file__).parent
    executable = script_dir / "test_spmm"

    if not executable.exists():
        print(f"{Colors.RED}错误: 可执行文件不存在{Colors.END}")
        print(f"请先运行: {Colors.YELLOW}python compile.py{Colors.END}\n")
        return 1

    print(f"{Colors.GREEN}可执行文件: {executable}{Colors.END}\n")

    # 创建测试器
    tester = SPMMTest(executable)

    # 运行测试
    start_time = time.time()

    try:
        run_boundary_tests(tester)
        run_gnn_tests(tester)
        run_n_multiplier_tests(tester)
        run_k_version_tests(tester)
        run_large_scale_tests(tester)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}测试被中断{Colors.END}")
        return 1

    # 打印总结
    success = tester.print_summary()

    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f} 秒")

    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}所有测试通过！✓{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}部分测试失败！{Colors.END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
