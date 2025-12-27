import torch

from .rode import RoDeCSR


def torch_spmm_reference(
    sparse_csr: torch.Tensor,
    dense_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch sparse SPMM 参考实现: C = A × B

    Args:
        sparse_csr: 稀疏矩阵 A (m, k) CSR 格式
        dense_matrix: 稠密矩阵 B (k, n)

    Returns:
        稠密矩阵 C (m, n)
    """
    # PyTorch 的 sparse_csr tensor 支持直接与 dense matrix 相乘
    return torch.sparse.mm(sparse_csr, dense_matrix)


def compare_results(
    rode_output: torch.Tensor,
    torch_output: torch.Tensor,
    tolerance: float = 1e-3,
) -> dict:
    """
    比较 RoDe 和 PyTorch sparse 的结果

    Args:
        rode_output: RoDe SPMM 输出 (m, n) dense tensor
        torch_output: torch.sparse SPMM 输出 (m, n) dense tensor
        tolerance: 误差容限

    Returns:
        包含比较结果的字典
    """
    # 确保都在 CPU 上
    rode_cpu = rode_output.cpu()
    torch_cpu = torch_output.cpu()

    # 检查形状
    if rode_cpu.shape != torch_cpu.shape:
        return {
            "passed": False,
            "error": f"Shape mismatch: RoDe={rode_cpu.shape}, Torch={torch_cpu.shape}",
            "mae": float("inf"),
            "mean_rel": float("inf"),
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "error_count": -1,
            "total_count": rode_cpu.numel(),
        }

    # 计算误差
    diff = torch.abs(rode_cpu - torch_cpu)
    rel_diff = diff / (torch.abs(torch_cpu) + 1e-8)

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
        "total_count": rode_cpu.numel(),
    }
