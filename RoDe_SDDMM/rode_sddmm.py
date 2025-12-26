"""
RoDe SDDMM Python 接口

提供 PyTorch 风格的 SDDMM 操作，支持与 torch.sparse 对比
"""

import math
from typing import Optional, Tuple

import torch


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
        m: int,
        n: int,
        k: int = 128,
        device: Optional[torch.device] = None,
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
        self.device = device or torch.device("cuda")

        # 确保数据类型正确
        self.row_offsets_cpu = row_offsets.to(dtype=torch.int32, device="cpu").contiguous()
        self.column_indices = column_indices.to(dtype=torch.int32, device=self.device).contiguous()
        self.values = values.to(dtype=torch.float32, device=self.device).contiguous()

        # 执行 RoDe 预处理
        self._preprocess()

        # 将 CSR 数据移动到 GPU
        self.row_offsets = self.row_offsets_cpu.to(device=self.device)

    def _preprocess(self):
        """执行 RoDe 预处理，生成 block 和 residue 索引"""
        try:
            # 尝试使用 CUDA 扩展
            import rode_sddmm_cuda

            config = self.CONFIGS[self.k]
            result = rode_sddmm_cuda.preprocess(
                self.row_offsets_cpu, self.nnz, config["seg_length"], config["k_block"], config["vec_len"]
            )

            block_indices, st_offsets, residue_indices, m1, m2 = result

            self.block_indices = block_indices.to(device=self.device)
            self.st_offsets = st_offsets.to(device=self.device)
            self.residue_indices = residue_indices.to(device=self.device)
            self.m1 = m1
            self.m2 = m2

        except ImportError:
            # 回退到纯 Python 实现
            self._preprocess_python()

    def _preprocess_python(self):
        """纯 Python 预处理实现（备用）"""
        config = self.CONFIGS[self.k]
        seg_length = config["seg_length"]
        k_block = config["k_block"]
        vec_len = config["vec_len"]

        row_ptr = self.row_offsets_cpu.numpy()

        block_r_ind = []
        st_off = []
        residue_r_ind = []

        for i in range(self.m):
            row_offset = int(row_ptr[i])
            n_padding = row_offset % vec_len
            row_nnz = int(row_ptr[i + 1]) - row_offset + n_padding

            if row_nnz > seg_length:
                block_r_ind.append(i)
                st_off.append(row_offset)
                row_offset = (row_offset + seg_length) - n_padding
                row_nnz -= seg_length

            while row_nnz > seg_length:
                block_r_ind.append(i)
                st_off.append(row_offset)
                row_offset += seg_length
                row_nnz -= seg_length

            if row_nnz > 0:
                if row_nnz >= k_block:
                    block_r_ind.append(i)
                    st_off.append(row_offset)
                if row_nnz % k_block:
                    residue_r_ind.append(i)

        st_off.append(int(row_ptr[self.m]))

        self.m1 = len(block_r_ind)
        self.m2 = len(residue_r_ind)

        # 创建 tensor
        self.block_indices = torch.tensor(
            block_r_ind if block_r_ind else [0], dtype=torch.int32, device=self.device
        )
        self.st_offsets = torch.tensor(st_off, dtype=torch.int32, device=self.device)
        self.residue_indices = torch.tensor(
            residue_r_ind if residue_r_ind else [0], dtype=torch.int32, device=self.device
        )

    @classmethod
    def from_dense(cls, dense_matrix: torch.Tensor, k: int = 128, pad_to: int = 4) -> "RoDeCSR":
        """
        从稠密矩阵创建 RoDe CSR 矩阵

        Args:
            dense_matrix: 稠密矩阵，形状 (m, n)
            k: 隐藏维度
            pad_to: 行填充对齐值

        Returns:
            RoDeCSR 实例
        """
        m, n = dense_matrix.shape
        device = dense_matrix.device

        # 转换为 CSR
        sparse_csr = dense_matrix.to_sparse_csr()

        row_offsets = sparse_csr.crow_indices().to(torch.int32)
        column_indices = sparse_csr.col_indices().to(torch.int32)
        values = sparse_csr.values().to(torch.float32)

        # 如果需要填充
        if pad_to > 1:
            row_offsets, column_indices, values = cls._pad_csr(
                row_offsets, column_indices, values, m, n, pad_to
            )

        return cls(row_offsets.cpu(), column_indices, values, m, n, k, device=device)

    @classmethod
    def from_scipy(
        cls, scipy_csr, k: int = 128, device: Optional[torch.device] = None, pad_to: int = 4
    ) -> "RoDeCSR":
        """
        从 scipy.sparse.csr_matrix 创建 RoDe CSR 矩阵

        Args:
            scipy_csr: scipy CSR 矩阵
            k: 隐藏维度
            device: 目标设备
            pad_to: 行填充对齐值

        Returns:
            RoDeCSR 实例
        """
        m, n = scipy_csr.shape
        device = device or torch.device("cuda")

        row_offsets = torch.from_numpy(scipy_csr.indptr.astype("int32"))
        column_indices = torch.from_numpy(scipy_csr.indices.astype("int32"))
        values = torch.from_numpy(scipy_csr.data.astype("float32"))

        # 如果需要填充
        if pad_to > 1:
            row_offsets, column_indices, values = cls._pad_csr(
                row_offsets, column_indices, values, m, n, pad_to
            )

        return cls(row_offsets, column_indices.to(device), values.to(device), m, n, k, device=device)

    @classmethod
    def from_random(
        cls,
        m: int,
        n: int,
        nnz_per_row: int,
        k: int = 128,
        device: Optional[torch.device] = None,
        seed: int = 42,
        pad_to: int = 4,
    ) -> "RoDeCSR":
        """
        生成随机 RoDe CSR 矩阵

        Args:
            m: 行数
            n: 列数
            nnz_per_row: 每行平均非零元素数
            k: 隐藏维度
            device: 目标设备
            seed: 随机种子
            pad_to: 行填充对齐值

        Returns:
            RoDeCSR 实例
        """
        import numpy as np

        device = device or torch.device("cuda")
        rng = np.random.RandomState(seed)

        row_offsets = [0]
        column_indices = []
        values = []

        for i in range(m):
            # 使用泊松分布生成每行 nnz
            row_nnz = max(1, min(rng.poisson(nnz_per_row), n))

            # 随机选择列索引
            cols = np.sort(rng.choice(n, row_nnz, replace=False))
            vals = rng.uniform(-1, 1, row_nnz).astype(np.float32)

            column_indices.extend(cols.tolist())
            values.extend(vals.tolist())

            # 填充对齐
            if pad_to > 1:
                residue = row_nnz % pad_to
                if residue > 0:
                    pad_count = pad_to - residue
                    column_indices.extend([int(cols[-1])] * pad_count)
                    values.extend([0.0] * pad_count)

            row_offsets.append(len(column_indices))

        return cls(
            torch.tensor(row_offsets, dtype=torch.int32),
            torch.tensor(column_indices, dtype=torch.int32, device=device),
            torch.tensor(values, dtype=torch.float32, device=device),
            m,
            n,
            k,
            device=device,
        )

    @staticmethod
    def _pad_csr(
        row_offsets: torch.Tensor,
        column_indices: torch.Tensor,
        values: torch.Tensor,
        m: int,
        n: int,
        pad_to: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对 CSR 矩阵的每行进行填充对齐
        """
        new_row_offsets = [0]
        new_column_indices = []
        new_values = []

        for i in range(m):
            start = int(row_offsets[i])
            end = int(row_offsets[i + 1])
            row_nnz = end - start

            # 复制原始数据
            for j in range(start, end):
                new_column_indices.append(int(column_indices[j]))
                new_values.append(float(values[j]))

            # 填充
            if row_nnz > 0 and pad_to > 1:
                residue = row_nnz % pad_to
                if residue > 0:
                    pad_count = pad_to - residue
                    last_col = int(column_indices[end - 1])
                    new_column_indices.extend([last_col] * pad_count)
                    new_values.extend([0.0] * pad_count)

            new_row_offsets.append(len(new_column_indices))

        return (
            torch.tensor(new_row_offsets, dtype=torch.int32),
            torch.tensor(new_column_indices, dtype=torch.int32, device=column_indices.device),
            torch.tensor(new_values, dtype=torch.float32, device=values.device),
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
            crow_indices, col_indices, values, size=(self.m, self.n), device=self.device
        )

    def __repr__(self) -> str:
        return (
            f"RoDeCSR(m={self.m}, n={self.n}, nnz={self.nnz}, k={self.k}, "
            f"m1={self.m1}, m2={self.m2}, device={self.device})"
        )


def rode_sddmm(sparse_matrix: RoDeCSR, lhs_matrix: torch.Tensor, rhs_matrix: torch.Tensor) -> torch.Tensor:
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

    if lhs_matrix.shape != (m, k):
        raise ValueError(f"lhs_matrix shape must be ({m}, {k}), got {lhs_matrix.shape}")
    if rhs_matrix.shape != (n, k):
        raise ValueError(f"rhs_matrix shape must be ({n}, {k}), got {rhs_matrix.shape}")

    # 确保数据类型和连续性
    lhs = lhs_matrix.to(dtype=torch.float32, device=sparse_matrix.device).contiguous()
    rhs = rhs_matrix.to(dtype=torch.float32, device=sparse_matrix.device).contiguous()

    try:
        # 使用 CUDA 扩展
        import rode_sddmm_cuda

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

    except ImportError:
        raise ImportError("rode_sddmm_cuda extension not found. Please compile with: python setup.py install")


def torch_sddmm_reference(
    sparse_csr: torch.Tensor, lhs_matrix: torch.Tensor, rhs_matrix: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch 参考实现的 SDDMM

    使用 torch.sparse.sampled_addmm API 实现 SDDMM:
        out = alpha * (mat1 @ mat2) * spy(input) + beta * input

    当 alpha=1, beta=0 时，等价于:
        out = (A × B^T) 在 S 非零位置的采样

    参考: https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html

    注意: sampled_addmm 计算 mat1 @ mat2（不是 mat1 @ mat2^T）
          因此需要传入 rhs_matrix.T 作为 mat2

    Args:
        sparse_csr: PyTorch sparse_csr tensor，形状 (m, n)
        lhs_matrix: 左侧稠密矩阵 (m, k)
        rhs_matrix: 右侧稠密矩阵 (n, k)

    Returns:
        SDDMM 结果的稀疏 CSR tensor
    """
    # 使用 torch.sparse.sampled_addmm
    # sampled_addmm(input, mat1, mat2, *, beta=1., alpha=1., out=None)
    # 计算: beta * input + alpha * (mat1 @ mat2) * spy(input)
    #
    # 对于 SDDMM，我们需要计算: S ⊙ (A × B^T)
    #   - input: 稀疏矩阵 S (用于提供稀疏模式)
    #   - mat1: 左侧稠密矩阵 A (m, k)
    #   - mat2: B^T，即 rhs_matrix.T (k, n)
    #   - alpha=1, beta=0: 只计算采样的矩阵乘法部分
    #
    # 因为 sampled_addmm 计算 mat1 @ mat2 (不是 mat1 @ mat2^T)
    # 所以我们需要先创建一个全1的稀疏矩阵来获取采样值，然后乘以原始稀疏值

    # 创建与 sparse_csr 结构相同但值全为1的稀疏矩阵
    ones_sparse = torch.sparse_csr_tensor(
        sparse_csr.crow_indices(),
        sparse_csr.col_indices(),
        torch.ones_like(sparse_csr.values()),
        size=sparse_csr.shape,
        device=sparse_csr.device,
        dtype=sparse_csr.dtype,
    )

    # 使用 sampled_addmm 计算采样的矩阵乘法
    # sampled_addmm 计算: beta * input + alpha * (mat1 @ mat2) * spy(input)
    # 其中 mat1 形状为 (m, k)，mat2 形状为 (k, n)
    # 我们需要计算 A @ B^T，其中 A (m, k)，B (n, k)
    # 因此 mat2 = B^T = rhs_matrix.T，形状为 (k, n)
    sampled_result = torch.sparse.sampled_addmm(
        ones_sparse,
        lhs_matrix,  # (m, k)
        rhs_matrix.T,  # (k, n)，转置后
        beta=0.0,
        alpha=1.0,
    )

    # 将采样结果乘以原始稀疏矩阵的值（Hadamard 积）
    result_values = sparse_csr.values() * sampled_result.values()

    # 重建 CSR 稀疏矩阵
    return torch.sparse_csr_tensor(
        sparse_csr.crow_indices(),
        sparse_csr.col_indices(),
        result_values,
        size=sparse_csr.shape,
        device=sparse_csr.device,
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
        }

    diff = torch.abs(rode_values - torch_values)
    rel_diff = diff / (torch.abs(torch_values) + 1e-6)

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
