import torch  # noqa

import rode_spmm_cuda


class RoDeCSR:
    """
    RoDe 格式的 CSR 稀疏矩阵封装

    在标准 CSR 格式基础上，预处理生成 RoDe 所需的元数据，
    用于高效执行 SPMM 操作。

    Attributes:
        m: 矩阵行数
        n: 矩阵列数 (稀疏矩阵的列数，即 k 维度)
        nnz: 非零元素数量
        k_version: k 版本 (32 或 128)，与 k 无关
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
    # 注意：k_version 与 k 无关，根据内部优化自动选择
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
        k_version: int = 128,
    ):
        """
        初始化 RoDe CSR 矩阵

        Args:
            row_offsets: CSR 行指针，形状 (m+1,)
            column_indices: CSR 列索引，形状 (nnz,)
            values: 非零值，形状 (nnz,)
            m: 矩阵行数
            n: 矩阵列数（对于 SPMM，这是稀疏矩阵的列数，即 k 维度）
            k_version: k 版本，必须是 32 或 128
                     注意：k_version 与 k 无关，根据内部优化自动选择
            device: 目标设备，默认 CUDA
        """
        if k_version not in self.CONFIGS:
            raise ValueError(f"k_version must be 32 or 128, got {k_version}")

        self.m = m
        self.n = n  # 对于 SPMM，这是稀疏矩阵的列数 k
        self.k_version = k_version
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
        config = self.CONFIGS[self.k_version]
        result = rode_spmm_cuda.preprocess(
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

    def __repr__(self) -> str:
        return (
            f"RoDeCSR(m={self.m}, n={self.n}, nnz={self.nnz}, k_version={self.k_version}, "
            f"m1={self.m1}, m2={self.m2}, device={self.device})"
        )

    # ========================================================
    # 以下RoDeCSR的成员方法应当用于测试场景，不应用于性能场景
    # ========================================================

    @classmethod
    def from_random(
        cls,
        m: int,
        n: int,
        nnz_per_row: int,
        device: torch.device,
        k_version: int = 128,
    ) -> "RoDeCSR":
        """
        创建随机稀疏矩阵

        Args:
            m: 行数
            n: 列数（对于 SPMM，这是稀疏矩阵的列数 k）
            nnz_per_row: 每行平均非零元素数
            device: 设备
            k_version: k 版本，默认 128
        """
        nnz_raw = m * nnz_per_row
        rows = torch.randint(0, m, (nnz_raw,))
        cols = torch.randint(0, n, (nnz_raw,))

        # 使用 torch.unique 对 (row, col) 进行去重
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
            k_version,
        )

    def to_torch_sparse_csr(self) -> torch.Tensor:
        """
        转换为 PyTorch sparse_csr tensor（不含填充）

        注意：
        1. 这会移除填充的零值
        2. 性能很差，不应用于性能场景
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


def rode_spmm(sparse_matrix: RoDeCSR, dense_matrix: torch.Tensor) -> torch.Tensor:
    """
    执行 RoDe SPMM: C = A × B

    其中：
    - A 是稀疏矩阵 (m, k) - sparse_matrix
    - B 是稠密矩阵 (k, n) - dense_matrix
    - C 是稠密矩阵 (m, n) - 输出

    Args:
        sparse_matrix: RoDe CSR 稀疏矩阵 A (m, k)
        dense_matrix: 稠密矩阵 B (k, n)

    Returns:
        输出稠密矩阵 C (m, n)
    """
    # 检查输入
    m, k = sparse_matrix.m, sparse_matrix.n  # 对于 SPMM，n 就是 k
    n = dense_matrix.size(1)  # 稠密矩阵的列数

    if dense_matrix.shape != (k, n):
        raise ValueError(f"dense_matrix shape must be ({k}, {n}), got {dense_matrix.shape}")

    assert dense_matrix.is_cuda and dense_matrix.device == sparse_matrix.device, (
        f"dense_matrix must be on {sparse_matrix.device}, got {dense_matrix.device}"
    )
    assert dense_matrix.dtype == torch.float32, f"dense_matrix must be float32, got {dense_matrix.dtype}"

    # 确保数据类型和连续性
    dense_matrix = dense_matrix.contiguous()

    # 检查 n 是否是 64 的倍数
    if n % 64 != 0:
        raise ValueError(f"n must be a multiple of 64, got {n}")

    out = rode_spmm_cuda.spmm_forward(
        sparse_matrix.block_indices,
        sparse_matrix.residue_indices,
        sparse_matrix.st_offsets,
        sparse_matrix.m1,
        sparse_matrix.m2,
        sparse_matrix.row_offsets,
        sparse_matrix.column_indices,
        sparse_matrix.values,
        m,
        k,
        n,
        dense_matrix,
    )
    return out
