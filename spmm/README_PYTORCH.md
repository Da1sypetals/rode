# RoDe SPMM PyTorch 扩展

## 概述

为 RoDe SPMM 实现了 PyTorch 扩展，可以方便地进行测试和验证。

## 编译

```bash
cd spmm
python setup.py build_ext --inplace
```

## 使用方法

### 基本用法

```python
import torch
from interface import RoDeCSR, rode_spmm, torch_spmm_reference, compare_results

# 创建 RoDe CSR 稀疏矩阵 A (m, k)
rode_csr = RoDeCSR.from_random(
    m=1024,
    k=512,
    nnz_per_row=32,
    device=torch.device('cuda:0'),
    k_version=128
)

# 创建稠密矩阵 B (k, n)
dense_matrix = torch.randn(512, 128, device='cuda', dtype=torch.float32)

# 执行 RoDe SPMM: C = A × B
output = rode_spmm(rode_csr, dense_matrix)
print(f"输出形状: {output.shape}")  # (1024, 128)
```

### 与 PyTorch sparse 对比

```python
# 转换为 PyTorch sparse CSR 格式
torch_sparse = rode_csr.to_torch_sparse_csr()

# 使用 PyTorch sparse 参考
torch_output = torch_spmm_reference(torch_sparse, dense_matrix)

# 比较结果
comparison = compare_results(output, torch_output)
print(f"MAE: {comparison['mae']:.2e}")
print(f"测试通过: {comparison['passed']}")
```

## 测试

运行所有测试：

```bash
python test_pytorch.py
```

按类别测试：

```bash
python test_pytorch.py --category small       # 小规模测试
python test_pytorch.py --category medium      # 中等规模测试
python test_pytorch.py --category large       # 大规模测试
python test_pytorch.py --category n_multiplier # n 的 64 倍数测试
python test_pytorch.py --category k_version   # k_version 测试
```

单项详细测试：

```bash
python test_pytorch.py --single 512 1024 64  # m=512, k=1024, nnz_per_row=64
```

列出所有类别：

```bash
python test_pytorch.py --list-categories
```

## 参数说明

### SPMM 接口: C = A × B

- **A**: CSR 格式稀疏矩阵，形状 (m, k)
- **B**: 稠密矩阵，形状 (k, n)
- **C**: 输出稠密矩阵，形状 (m, n)

### 参数限制

1. **m** (稀疏矩阵行数): 无限制
   - 支持任意正整数
   - 典型值: 64, 128, 256, 512, 1024, 4096, 8192, 16384, 65536

2. **k** (稀疏矩阵列数 / 稠密矩阵行数): 无限制
   - 支持任意正整数
   - 典型值: 64, 128, 256, 512, 1024

3. **n** (稠密矩阵列数 / 输出维度): **必须是 64 的倍数**
   - 限制: n % 64 == 0
   - 有效值: 64, 128, 192, 256, 320, 384, 448, 512, ...
   - 无效值: 1-63, 65-127, 129-191, 193-255, ...

4. **k_version** (内部优化参数): 与 k 无关
   - 值: 32 或 128
   - 说明: 自动选择，无需手动指定
   - k <= 32: 使用 k_version=32
   - k > 32: 使用 k_version=128
   - 影响: 仅影响性能，不影响正确性

## 测试类别

| 类别 | 描述 | 测试数 |
|------|------|--------|
| small | 小规模测试 | 5 |
| irregular | Irregular 维度测试 | 3 |
| medium | 中等规模测试 | 5 |
| rectangular | 非方阵测试 | 3 |
| sparsity | 稀疏程度测试 | 3 |
| prime | 素数维度测试 | 2 |
| n_multiplier | n 的 64 倍数测试 | 4 |
| k_version | k_version 自动选择测试 | 4 |
| large | 大规模测试 | 4 |

**总计**: 33 个测试

## 测试结果

所有测试通过：
- MAE (平均绝对误差): < 1e-6
- 最大相对误差: < 1e-4
- 通过率: 100% (33/33)

## 文件结构

```
spmm/
├── setup.py                    # PyTorch 扩展编译脚本
├── csrc/
│   ├── rode_spmm_cuda.cu      # PyTorch CUDA 绑定
│   ├── RoDeSpmm.cu           # RoDe SPMM CUDA kernel
│   ├── RoDeSpmm.h            # 函数声明
│   ├── basic_utils.h          # 基础工具
│   └── common_utils.h         # 通用工具
├── interface/
│   ├── __init__.py           # 模块导出
│   ├── rode.py               # RoDeCSR 类和 rode_spmm 函数
│   └── utils.py             # 工具函数 (compare_results, torch_spmm_reference)
├── test_pytorch.py           # PyTorch 测试脚本
└── test_quick.py             # 快速测试脚本
```

## 性能说明

注意：当前实现仅用于测试和验证，性能场景下的使用需要进一步优化。

测试结果显示，在某些情况下 RoDe SPMM 相比 torch.sparse 可以获得更好的性能（加速比可达 8.66x），但这取决于具体的矩阵规模和稀疏程度。
