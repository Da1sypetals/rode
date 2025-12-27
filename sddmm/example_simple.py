#!/usr/bin/env python3
"""
RoDe SDDMM 最简示例
生成随机CSR矩阵，执行SDDMM计算，并打印前10个值（按COO顺序）
"""

import torch
from interface import RoDeCSR, rode_sddmm, torch_sddmm_reference

# 设置参数
m, n, k = 128, 256, 128
nnz_per_row = 32
device = torch.device("cuda:0")
dtype = torch.float32

print("=== RoDe SDDMM 示例 ===")
print(f"矩阵尺寸: m={m}, n={n}, k={k}, nnz_per_row={nnz_per_row}")
print()

# ============================================
# 1. 生成随机 COO 稀疏矩阵并去重，转换为 CSR
# ============================================

# 生成随机 COO 坐标（可能有重复）
nnz_raw = m * nnz_per_row
rows = torch.randint(0, m, (nnz_raw,))
cols = torch.randint(0, n, (nnz_raw,))

# 使用 torch.unique 对 (row, col) 进行去重
# 将 (row, col) 编码为单个值进行去重
coords = rows * n + cols
unique_coords, inverse_indices = torch.unique(coords, return_inverse=True)

# 解码回 (row, col)
unique_rows = unique_coords // n
unique_cols = unique_coords % n

# 生成去重后的随机值
nnz = unique_coords.shape[0]
unique_values = torch.randn(nnz, dtype=dtype)

# 使用 torch.sparse_coo_tensor 创建 COO，然后转换为 CSR
indices = torch.stack([unique_rows, unique_cols], dim=0)
coo_sparse = torch.sparse_coo_tensor(indices, unique_values, size=(m, n))
csr_sparse = coo_sparse.to_sparse_csr()

# 提取 CSR 组件
row_offsets = csr_sparse.crow_indices().to(torch.int32)
col_indices = csr_sparse.col_indices().to(torch.int32).to(device)
values = csr_sparse.values().to(dtype).to(device)

nnz = values.shape[0]
print(f"稀疏矩阵 S: shape=({m}, {n}), nnz={nnz}")

# ============================================
# 2. 创建 RoDeCSR 矩阵（会自动进行预处理和填充）
# ============================================
rode_csr = RoDeCSR(
    row_offsets=row_offsets,
    column_indices=col_indices,
    values=values,
    device=device,
    m=m,
    n=n,
    k=k,
)

# ============================================
# 3. 生成随机稠密矩阵 A 和 B
# ============================================
A = torch.randn(m, k, dtype=dtype, device=device)  # (m, k)
B = torch.randn(n, k, dtype=dtype, device=device)  # (n, k)

print(f"稠密矩阵 A: shape={A.shape}")
print(f"稠密矩阵 B: shape={B.shape}")
print()

# ============================================
# 4. 使用 RoDe SDDMM 计算: out = S ⊙ (A × B^T)
# ============================================
rode_output = rode_sddmm(rode_csr, A, B)

print("=== RoDe SDDMM 结果（前10个值，按COO顺序）===")

# 使用 tensor 操作获取 COO 格式的 (row, col, value)
# 从 row_offsets 生成每个元素的行索引
row_offsets_cpu = rode_csr.row_offsets.cpu()
col_indices_cpu = rode_csr.column_indices.cpu()
values_cpu = rode_csr.values.cpu()
output_cpu = rode_output.cpu()

# 计算每行的非零元素数量
row_counts = row_offsets_cpu[1:] - row_offsets_cpu[:-1]
# 生成行索引（每个非零元素对应的行号）
row_indices = torch.repeat_interleave(torch.arange(m), row_counts)

# 找到非填充元素的掩码（原始非零位置）
# RoDeCSR 会填充，填充值为0，但原始值也可能为0，所以用原始nnz数量
original_nnz = nnz
non_pad_mask = torch.arange(row_indices.shape[0]) < original_nnz

# 提取前10个非填充元素
top_k = 10
top_rows = row_indices[:top_k]
top_cols = col_indices_cpu[:top_k]
top_vals = values_cpu[:top_k]
top_outs = output_cpu[:top_k]

# 打印前10个
for i in range(min(top_k, nnz)):
    print(
        f"coord = ({top_rows[i].item()}, {top_cols[i].item()}), val = {top_vals[i].item():+7.4f}, out = {top_outs[i].item():+10.4f}"
    )

# ============================================
# 5. 与 PyTorch 参考实现对比验证
# ============================================
print()
print("=== 与 torch.sparse.sampled_addmm 对比验证 ===")

# 将 RoDeCSR 转换为 PyTorch sparse_csr（不含填充）
torch_csr = rode_csr.to_torch_sparse_csr()

# PyTorch 参考实现
torch_result = torch_sddmm_reference(torch_csr, A, B)
torch_output = torch_result.values()

# 使用 tensor 操作提取 RoDe 的非填充值进行对比
# RoDeCSR 填充后的输出，取前 original_nnz 个值
rode_values_no_pad = output_cpu[:original_nnz]
torch_output_cpu = torch_output.cpu()

# 计算误差（全部使用 tensor 操作）
diff = torch.abs(rode_values_no_pad - torch_output_cpu)
mae = diff.mean().item()
max_abs = diff.max().item()
rel_diff = diff / (torch.abs(torch_output_cpu) + 1e-8)
max_rel = rel_diff.max().item()

print(f"平均绝对误差 (MAE): {mae:.6e}")
print(f"最大绝对误差: {max_abs:.6e}")
print(f"最大相对误差: {max_rel:.6e}")
print(f"验证结果: {'✓ PASS' if max_abs < 1e-3 else '✗ FAIL'}")
