/**
 * RoDe SPMM PyTorch CUDA 扩展
 *
 * 提供 PyTorch tensor 接口来调用 RoDe SPMM CUDA kernel
 * SPMM: C = A × B，其中 A 是 CSR 稀疏矩阵 (m, k)，B 是稠密矩阵 (k, n)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// 声明 RoDe SPMM 函数 (来自 RoDeSpmm.cu)
extern void RoDeSpmm_n32(
    int m1, int m2, int k, int n,
    const float* __restrict__ values,
    const int* __restrict__ column_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ row_indices1,
    const int* __restrict__ row_indices2,
    const int* __restrict__ row_seg_st_offsets,
    const float* B,
    float* C,
    cudaStream_t stream1,
    cudaStream_t stream2
);

extern void RoDeSpmm_n128(
    int m1, int m2, int k, int n,
    const float* __restrict__ values,
    const int* __restrict__ column_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ row_indices1,
    const int* __restrict__ row_indices2,
    const int* __restrict__ row_seg_st_offsets,
    const float* B,
    float* C,
    cudaStream_t stream1,
    cudaStream_t stream2
);

// ============================================================================
// 宏定义
// ============================================================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_INT(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")

// ============================================================================
// RoDe 预处理函数 (CPU)
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int> rode_preprocess(
    torch::Tensor row_offsets,  // CSR 行指针 (m+1,)
    int nnz,                     // 非零元素数
    int seg_length,              // 段长度
    int k_block,                 // k 块大小：32
    int vec_len                  // 向量长度：4
) {
    // 检查输入
    TORCH_CHECK(row_offsets.device().is_cpu(), "row_offsets must be on CPU for preprocessing");
    TORCH_CHECK(row_offsets.dtype() == torch::kInt32, "row_offsets must be int32");

    int m = row_offsets.size(0) - 1;
    const int* row_ptr = row_offsets.data_ptr<int>();

    // 预估最大 block 数量
    int max_blocks = nnz / seg_length + m + 100;

    // 分配临时存储
    std::vector<int> block_r_ind(max_blocks);
    std::vector<int> st_off(max_blocks + 1);
    std::vector<int> residue_r_ind(m);

    int n_blk = 0;
    int n_res = 0;

    // 预处理逻辑
    for (int i = 0; i < m; ++i) {
        int row_offset = row_ptr[i];
        int n_padding = row_offset % vec_len;
        int row_nnz = row_ptr[i + 1] - row_offset + n_padding;

        if (row_nnz > seg_length) {
            block_r_ind[n_blk] = i;
            st_off[n_blk++] = row_offset;
            row_offset = (row_offset + seg_length) - n_padding;
            row_nnz -= seg_length;
        }

        while (row_nnz > seg_length) {
            block_r_ind[n_blk] = i;
            st_off[n_blk++] = row_offset;
            row_offset += seg_length;
            row_nnz -= seg_length;
        }

        if (row_nnz > 0) {
            if (row_nnz >= k_block) {
                block_r_ind[n_blk] = i;
                st_off[n_blk++] = row_offset;
            }
            if (row_nnz % k_block) {
                residue_r_ind[n_res++] = i;
            }
        }
    }

    st_off[n_blk] = row_ptr[m];

    // 创建输出 tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32);

    torch::Tensor block_indices = torch::empty({std::max(n_blk, 1)}, options);
    torch::Tensor st_offsets_tensor = torch::empty({n_blk + 1}, options);
    torch::Tensor residue_indices = torch::empty({std::max(n_res, 1)}, options);

    if (n_blk > 0) {
        memcpy(block_indices.data_ptr<int>(), block_r_ind.data(), n_blk * sizeof(int));
        memcpy(st_offsets_tensor.data_ptr<int>(), st_off.data(), (n_blk + 1) * sizeof(int));
    } else {
        st_offsets_tensor.index_put_({0}, row_ptr[m]);
    }

    if (n_res > 0) {
        memcpy(residue_indices.data_ptr<int>(), residue_r_ind.data(), n_res * sizeof(int));
    }

    return std::make_tuple(block_indices, st_offsets_tensor, residue_indices, n_blk, n_res);
}

// ============================================================================
// RoDe SPMM 前向传播
// ============================================================================
torch::Tensor rode_spmm_forward(
    // RoDe 元数据
    torch::Tensor block_indices,    // Block 部分行索引
    torch::Tensor residue_indices,  // Residue 部分行索引
    torch::Tensor st_offsets,       // 起始偏移量
    int m1,                          // Block 部分数量
    int m2,                          // Residue 部分数量
    // CSR 矩阵数据
    torch::Tensor row_offsets,      // 行偏移量
    torch::Tensor column_indices,   // 列索引
    torch::Tensor values,           // 稀疏值
    int m,                          // 行数
    int k,                          // 稀疏矩阵列数
    int n,                          // 稠密矩阵列数
    // 稠密矩阵
    torch::Tensor dense_matrix      // 稠密矩阵 B (k, n)
) {
    // 检查输入
    CHECK_INPUT(block_indices);
    CHECK_INPUT(residue_indices);
    CHECK_INPUT(st_offsets);
    CHECK_INPUT(row_offsets);
    CHECK_INPUT(column_indices);
    CHECK_INPUT(values);
    CHECK_INPUT(dense_matrix);

    CHECK_INT(block_indices);
    CHECK_INT(residue_indices);
    CHECK_INT(st_offsets);
    CHECK_INT(row_offsets);
    CHECK_INT(column_indices);
    CHECK_FLOAT(values);
    CHECK_FLOAT(dense_matrix);

    TORCH_CHECK(dense_matrix.size(0) == k, "dense_matrix rows must equal k");
    TORCH_CHECK(dense_matrix.size(1) == n, "dense_matrix cols must equal n");

    // 创建输出 tensor C (m, n)
    auto out = torch::zeros({m, n}, dense_matrix.options());

    // 如果 m1 和 m2 都为 0，则直接返回全零结果
    if (m1 == 0 && m2 == 0) {
        return out;
    }

    // 获取数据指针
    const int* d_block_indices = block_indices.data_ptr<int>();
    const int* d_residue_indices = residue_indices.data_ptr<int>();
    const int* d_st_offsets = st_offsets.data_ptr<int>();
    const int* d_row_offsets = row_offsets.data_ptr<int>();
    const int* d_column_indices = column_indices.data_ptr<int>();
    const float* d_values = values.data_ptr<float>();
    const float* d_dense_matrix = dense_matrix.data_ptr<float>();
    float* d_out = out.data_ptr<float>();

    // 创建 CUDA 流
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;

    if (m1 > 0) {
        cudaStreamCreate(&stream1);
    }
    if (m2 > 0) {
        cudaStreamCreate(&stream2);
    }

    // 根据 n 选择版本：n < 64 使用 n32，n >= 64 使用 n128
    // 注意：n 必须是 64 的倍数
    if (n < 64) {
        if (m1 > 0 || m2 > 0) {
            RoDeSpmm_n32(
                m1, m2, k, n,
                d_values,
                d_column_indices,
                d_row_offsets,
                d_block_indices,
                d_residue_indices,
                d_st_offsets,
                d_dense_matrix,
                d_out,
                stream1 ? stream1 : 0,
                stream2 ? stream2 : 0
            );
        }
    } else {
        if (m1 > 0 || m2 > 0) {
            RoDeSpmm_n128(
                m1, m2, k, n,
                d_values,
                d_column_indices,
                d_row_offsets,
                d_block_indices,
                d_residue_indices,
                d_st_offsets,
                d_dense_matrix,
                d_out,
                stream1 ? stream1 : 0,
                stream2 ? stream2 : 0
            );
        }
    }

    // 同步流
    if (stream1) {
        cudaStreamSynchronize(stream1);
        cudaStreamDestroy(stream1);
    }
    if (stream2) {
        cudaStreamSynchronize(stream2);
        cudaStreamDestroy(stream2);
    }

    return out;
}

// ============================================================================
// PyBind11 模块定义
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &rode_preprocess,
          "RoDe CSR preprocessing (CPU)",
          py::arg("row_offsets"),
          py::arg("nnz"),
          py::arg("seg_length"),
          py::arg("k_block"),
          py::arg("vec_len"));

    m.def("spmm_forward", &rode_spmm_forward,
          "RoDe SPMM forward (CUDA): C = A × B",
          py::arg("block_indices"),
          py::arg("residue_indices"),
          py::arg("st_offsets"),
          py::arg("m1"),
          py::arg("m2"),
          py::arg("row_offsets"),
          py::arg("column_indices"),
          py::arg("values"),
          py::arg("m"),
          py::arg("k"),
          py::arg("n"),
          py::arg("dense_matrix"));
}
