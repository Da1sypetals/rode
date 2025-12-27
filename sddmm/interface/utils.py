import torch

from .rode import RoDeCSR


def torch_sddmm_reference(
    sparse_csr: torch.Tensor,
    lhs_matrix: torch.Tensor,
    rhs_matrix: torch.Tensor,
) -> torch.Tensor:
    # sparse_csr 的值不参与mm，只使用其稀疏模式(sparsity pattern)
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
