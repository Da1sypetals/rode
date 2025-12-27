import torch

from .rode import RoDeCSR


def torch_sddmm_reference(
    sparse_csr: torch.Tensor,
    lhs_matrix: torch.Tensor,
    rhs_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch 参考实现的 SDDMM: out = S ⊙ (A × B^T)

    使用 torch.sparse.sampled_addmm API 实现 SDDMM:
        sampled_addmm(input, mat1, mat2, beta, alpha) = alpha * (mat1 @ mat2) * spy(input) + beta * input

    当 beta=1.0, alpha=1.0 时:
        out = (mat1 @ mat2) * spy(input) + input
            = (A × B^T) 采样值 + S 原始值

    但我们需要的是 S ⊙ (A × B^T)，即 S 的值 × 采样值。

    最高效实现：直接传入 sparse_csr，设置 beta=0.0
        - sampled_addmm 只使用 sparse_csr 的稀疏模式进行采样
        - 返回 (A × B^T) 在 S 非零位置的采样值
        - 然后乘以 S 的原始值得到最终 SDDMM 结果

    参考: https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html

    Args:
        sparse_csr: PyTorch sparse_csr tensor，形状 (m, n)
        lhs_matrix: 左侧稠密矩阵 (m, k)
        rhs_matrix: 右侧稠密矩阵 (n, k)

    Returns:
        SDDMM 结果的稀疏 CSR tensor
    """
    # 最高效实现：直接使用 sparse_csr，避免创建额外的 ones_sparse
    # beta=0.0 时，sparse_csr 的值不参与加法，只使用其稀疏模式
    sampled = torch.sparse.sampled_addmm(
        sparse_csr,
        lhs_matrix,  # (m, k)
        rhs_matrix.T,  # (k, n)
        beta=0.0,
        alpha=1.0,
    )

    # SDDMM = S ⊙ (A × B^T)，需要乘以 S 的原始值
    # 构造最终结果：采样值 × S 的值
    result_values = sampled.values() * sparse_csr.values()

    return torch.sparse_csr_tensor(
        sampled.crow_indices(),
        sampled.col_indices(),
        result_values,
        size=sampled.shape,
        device=sampled.device,
        dtype=sampled.dtype,
    )


def compare_results(
    rode_output: torch.Tensor,
    torch_sparse_output: torch.Tensor,
    rode_csr: RoDeCSR,
    tolerance: float = 1e-3,
) -> dict:
    """
    比较 RoDe 和 PyTorch sparse 的结果

    Args:
        rode_output: RoDe SDDMM 输出 (nnz,)
        torch_sparse_output: torch.sparse SDDMM 输出 (sparse CSR tensor)
        rode_csr: RoDe CSR 矩阵
        tolerance: 误差容限

    Returns:
        包含比较结果的字典
    """
    # 获取 torch sparse 的值
    torch_values = torch_sparse_output.values()

    # RoDe 输出需要移除填充位置
    rode_values = []
    torch_idx = 0

    row_offsets = rode_csr.row_offsets.cpu()
    values_cpu = rode_csr.values.cpu()
    rode_out_cpu = rode_output.cpu()

    for i in range(rode_csr.m):
        start = int(row_offsets[i])
        end = int(row_offsets[i + 1])

        for j in range(start, end):
            if values_cpu[j] != 0:  # 非填充位置
                rode_values.append(float(rode_out_cpu[j]))

    rode_values = torch.tensor(rode_values, dtype=torch.float32)
    torch_values = torch_values.cpu().to(torch.float32)

    # 计算误差
    if rode_values.numel() != torch_values.numel():
        return {
            "passed": False,
            "error": f"Size mismatch: RoDe={rode_values.numel()}, Torch={torch_values.numel()}",
            "mae": float("inf"),
            "mean_rel": float("inf"),
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "error_count": -1,
            "total_count": rode_values.numel(),
        }

    diff = torch.abs(rode_values - torch_values)
    rel_diff = diff / (torch.abs(torch_values) + 1e-8)

    mae = diff.mean().item()
    mean_rel = rel_diff.mean().item()
    max_abs = diff.max().item()
    max_rel = rel_diff.max().item()

    error_mask = (diff > tolerance) & (rel_diff > tolerance)
    error_count = error_mask.sum().item()

    passed = error_count == 0

    return {
        "passed": passed,
        "mae": mae,
        "mean_rel": mean_rel,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "error_count": error_count,
        "total_count": rode_values.numel(),
    }
