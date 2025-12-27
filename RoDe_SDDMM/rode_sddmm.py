"""
RoDe SDDMM Python 接口

提供 PyTorch 风格的 SDDMM 操作，支持与 torch.sparse 对比
"""

import torch  # noqa

import rode_sddmm_cuda


class RoDeCSR:
    """
    RoDe 格式的 CSR 稀疏矩阵封装

    在标准 CSR 格式基础上，预处理生成 RoDe 所需的元数据，
    用于高效执行 SDDMM 操作。

    Attributes:
        m: 矩阵行数
        n: 矩阵列数
        nnz: 非零元素数量
        k: 隐藏维度 (32 或 128)
        row_offsets: CSR 行指针 (m+1,)
        column_indices: CSR 列索引 (nnz,)
        values: CSR 非零值 (nnz,)
        block_indices: Block 部分行索引
        residue_indices: Residue 部分行索引
        st_offsets: Block 部分起始偏移
        m1: Block 部分数量
        m2: Residue 部分数量
    """

    # RoDe 参数配置
    CONFIGS = {
        32: {"seg_length": 512, "k_block": 32, "vec_len": 4},
        128: {"seg_length": 32, "k_block": 32, "vec_len": 4},
    }

    def __init__(
        self,
        row_offsets: torch.Tensor,
        column_indices: torch.Tensor,
        values: torch.Tensor,
        device: torch.device,
        m: int,
        n: int,
        k: int = 128,
    ):
        """
        初始化 RoDe CSR 矩阵

        Args:
            row_offsets: CSR 行指针，形状 (m+1,)
            column_indices: CSR 列索引，形状 (nnz,)
            values: 非零值，形状 (nnz,)
            m: 矩阵行数
            n: 矩阵列数
            k: 隐藏维度，必须是 32 或 128
            device: 目标设备，默认 CUDA
        """
        if k not in self.CONFIGS:
            raise ValueError(f"k must be 32 or 128, got {k}")

        self.m = m
        self.n = n
        self.k = k
        self.nnz = values.numel()
        self.device = torch.device(device)

        # 确保数据类型正确
        self.row_offsets_cpu = row_offsets.to(dtype=torch.int32, device="cpu").contiguous()
        self.column_indices = column_indices.to(dtype=torch.int32, device=self.device).contiguous()
        self.values = values.to(dtype=torch.float32, device=self.device).contiguous()

        # 执行 RoDe 预处理
        self._preprocess()

        # 将 CSR 数据移动到 GPU
        self.row_offsets = self.row_offsets_cpu.to(device=self.device)

    def _preprocess(self):
        config = self.CONFIGS[self.k]
        result = rode_sddmm_cuda.preprocess(
            self.row_offsets_cpu,
            self.nnz,
            config["seg_length"],
            config["k_block"],
            config["vec_len"],
        )

        block_indices, st_offsets, residue_indices, m1, m2 = result

        self.block_indices = block_indices.to(device=self.device)
        self.st_offsets = st_offsets.to(device=self.device)
        self.residue_indices = residue_indices.to(device=self.device)
        self.m1 = m1
        self.m2 = m2

    @classmethod
    def from_random(
        cls,
        m: int,
        n: int,
        nnz_per_row: int,
        device: torch.device,
        k: int = 128,
    ) -> "RoDeCSR":
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
        unique_values = torch.randn(nnz, dtype=torch.float32)

        # 使用 torch.sparse_coo_tensor 创建 COO，然后转换为 CSR
        indices = torch.stack([unique_rows, unique_cols], dim=0)
        coo_sparse = torch.sparse_coo_tensor(indices, unique_values, size=(m, n))
        csr_sparse = coo_sparse.to_sparse_csr()

        # 提取 CSR 组件
        row_offsets = csr_sparse.crow_indices().to(torch.int32)
        col_indices = csr_sparse.col_indices().to(torch.int32).to(device)
        values = csr_sparse.values().to(torch.float32).to(device)

        return cls(
            row_offsets,
            col_indices,
            values,
            device,
            m,
            n,
            k,
        )

    def to_torch_sparse_csr(self) -> torch.Tensor:
        """
        转换为 PyTorch sparse_csr tensor（不含填充）

        注意：这会移除填充的零值
        """
        # 移除填充（值为0的元素）
        mask = self.values != 0

        # 重建不含填充的 CSR
        new_row_offsets = [0]
        new_col_indices = []
        new_values = []

        row_offsets_cpu = self.row_offsets.cpu()
        col_indices_cpu = self.column_indices.cpu()
        values_cpu = self.values.cpu()

        for i in range(self.m):
            start = int(row_offsets_cpu[i])
            end = int(row_offsets_cpu[i + 1])

            for j in range(start, end):
                if values_cpu[j] != 0:
                    new_col_indices.append(int(col_indices_cpu[j]))
                    new_values.append(float(values_cpu[j]))

            new_row_offsets.append(len(new_col_indices))

        crow_indices = torch.tensor(new_row_offsets, dtype=torch.int64)
        col_indices = torch.tensor(new_col_indices, dtype=torch.int64)
        values = torch.tensor(new_values, dtype=torch.float32)

        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=(self.m, self.n),
            device=self.device,
        )

    def __repr__(self) -> str:
        return (
            f"RoDeCSR(m={self.m}, n={self.n}, nnz={self.nnz}, k={self.k}, "
            f"m1={self.m1}, m2={self.m2}, device={self.device})"
        )


def rode_sddmm(sparse_matrix: RoDeCSR, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    执行 RoDe SDDMM: out = S ⊙ (A × B^T)

    其中：
    - S 是稀疏矩阵 (m, n)
    - A 是左侧稠密矩阵 (m, k)
    - B 是右侧稠密矩阵 (n, k)
    - ⊙ 是 Hadamard 积（逐元素乘法），仅在 S 的非零位置采样

    Args:
        sparse_matrix: RoDe CSR 稀疏矩阵
        lhs_matrix: 左侧稠密矩阵，形状 (m, k)
        rhs_matrix: 右侧稠密矩阵，形状 (n, k)

    Returns:
        输出稀疏矩阵的值，形状 (nnz,)
    """
    # 检查输入
    m, n, k = sparse_matrix.m, sparse_matrix.n, sparse_matrix.k

    if lhs.shape != (m, k):
        raise ValueError(f"lhs_matrix shape must be ({m}, {k}), got {lhs.shape}")
    if rhs.shape != (n, k):
        raise ValueError(f"rhs_matrix shape must be ({n}, {k}), got {rhs.shape}")

    assert lhs.is_cuda and lhs.device == sparse_matrix.device, (
        f"lhs must be on {sparse_matrix.device}, got {lhs.device}"
    )
    assert rhs.is_cuda and rhs.device == sparse_matrix.device, (
        f"rhs must be on {sparse_matrix.device}, got {rhs.device}"
    )
    assert lhs.dtype == torch.float32, f"lhs must be float32, got {lhs.dtype}"
    assert rhs.dtype == torch.float32, f"rhs must be float32, got {rhs.dtype}"

    # 确保数据类型和连续性
    lhs = lhs.contiguous()
    rhs = rhs.contiguous()

    out = rode_sddmm_cuda.sddmm_forward(
        sparse_matrix.block_indices,
        sparse_matrix.residue_indices,
        sparse_matrix.st_offsets,
        sparse_matrix.m1,
        sparse_matrix.m2,
        sparse_matrix.row_offsets,
        sparse_matrix.column_indices,
        sparse_matrix.values,
        m,
        n,
        lhs,
        rhs,
        k,
    )
    return out


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
    rode_output: torch.Tensor, torch_sparse_output: torch.Tensor, rode_csr: RoDeCSR, tolerance: float = 1e-3
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
