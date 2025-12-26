#!/bin/bash

# RoDe SDDMM 编译脚本
# 
# 用法:
#   ./compile.sh          # 自动检测GPU架构并编译
#   ./compile.sh sm_75    # 指定GPU架构编译

set -e

# 默认编译参数
OUTPUT="test_sddmm"
SOURCES="main.cu RoDeSddmm.cu"
INCLUDE_DIR="."
STD="c++17"
OPT_LEVEL="-O3"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检测GPU架构
detect_arch() {
    if command -v nvidia-smi &> /dev/null; then
        # 尝试从nvidia-smi获取计算能力
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [ -n "$COMPUTE_CAP" ]; then
            echo "sm_${COMPUTE_CAP}"
            return
        fi
    fi
    
    # 默认使用sm_75 (Tesla T4)
    echo "sm_75"
}

# 获取架构参数
if [ -n "$1" ]; then
    ARCH="$1"
else
    ARCH=$(detect_arch)
fi

echo "============================================"
echo "RoDe SDDMM 编译"
echo "============================================"
echo "GPU架构: $ARCH"
echo "输出文件: $OUTPUT"
echo "源文件: $SOURCES"
echo "============================================"

# 编译命令
NVCC_CMD="nvcc $OPT_LEVEL -arch=$ARCH -std=$STD -o $OUTPUT $SOURCES -I$INCLUDE_DIR"

echo "执行编译命令:"
echo "  $NVCC_CMD"
echo ""

# 执行编译
$NVCC_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ 编译成功!"
    echo "============================================"
    echo ""
    echo "运行方式:"
    echo "  ./test_sddmm [m] [n] [k] [nnz_per_row]"
    echo ""
    echo "参数说明:"
    echo "  m: 稀疏矩阵行数 (默认: 1024)"
    echo "  n: 稀疏矩阵列数 (默认: 2048)"
    echo "  k: 隐藏维度，必须是 32 或 128 (默认: 128)"
    echo "  nnz_per_row: 每行平均非零元素数 (默认: 64)"
    echo ""
    echo "示例:"
    echo "  ./test_sddmm 1024 2048 128 64    # k=128 版本"
    echo "  ./test_sddmm 2048 4096 32 100    # k=32 版本"
else
    echo ""
    echo "============================================"
    echo "✗ 编译失败!"
    echo "============================================"
    exit 1
fi
