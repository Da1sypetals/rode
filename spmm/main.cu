#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#include "RoDeSpmm.h"

using namespace std;

// ============================================================================
// 宏定义和错误检查
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// CPU 参考实现 - SPMM: C = A × B
// ============================================================================
void cpu_spmm_reference(
    const int* row_offsets,
    const int* column_indices,
    const float* values,
    const float* B,
    float* C,
    int m, int k, int n
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0.0f;
        }

        int row_start = row_offsets[i];
        int row_end = row_offsets[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = column_indices[idx];
            float val = values[idx];

            for (int j = 0; j < n; ++j) {
                C[i * n + j] += val * B[col * n + j];
            }
        }
    }
}

// ============================================================================
// RoDe 预处理函数 (CPU) - 直接复用 SDDMM 的逻辑
// ============================================================================
struct RoDeMetadata {
    std::vector<int> block_indices;
    std::vector<int> residue_indices;
    std::vector<int> st_offsets;
    int m1;
    int m2;

    RoDeMetadata(int max_blocks, int m)
        : block_indices(max_blocks), residue_indices(m), st_offsets(max_blocks + 1), m1(0), m2(0) {}
};

RoDeMetadata rode_preprocess(
    const int* row_offsets,
    int nnz,
    int m,
    int seg_length,
    int k_block,
    int vec_len
) {
    RoDeMetadata meta(nnz / seg_length + m + 100, m);
    int n_blk = 0;
    int n_res = 0;

    for (int i = 0; i < m; ++i) {
        int row_offset = row_offsets[i];
        int n_padding = row_offset % vec_len;
        int row_nnz = row_offsets[i + 1] - row_offset + n_padding;

        if (row_nnz > seg_length) {
            meta.block_indices[n_blk] = i;
            meta.st_offsets[n_blk++] = row_offset;
            row_offset = (row_offset + seg_length) - n_padding;
            row_nnz -= seg_length;
        }

        while (row_nnz > seg_length) {
            meta.block_indices[n_blk] = i;
            meta.st_offsets[n_blk++] = row_offset;
            row_offset += seg_length;
            row_nnz -= seg_length;
        }

        if (row_nnz > 0) {
            if (row_nnz >= k_block) {
                meta.block_indices[n_blk] = i;
                meta.st_offsets[n_blk++] = row_offset;
            }
            if (row_nnz % k_block) {
                meta.residue_indices[n_res++] = i;
            }
        }
    }

    meta.st_offsets[n_blk] = row_offsets[m];
    meta.m1 = n_blk;
    meta.m2 = n_res;

    meta.block_indices.resize(std::max(n_blk, 1));
    meta.st_offsets.resize(n_blk + 1);
    meta.residue_indices.resize(std::max(n_res, 1));

    return meta;
}

// ============================================================================
// 比较结果
// ============================================================================
bool compare_results(
    const float* gpu_result,
    const float* cpu_result,
    int m, int n,
    float tolerance = 1e-3
) {
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    int error_count = 0;

    for (int i = 0; i < m * n; ++i) {
        float abs_error = std::abs(gpu_result[i] - cpu_result[i]);
        float rel_error = abs_error / (std::abs(cpu_result[i]) + 1e-8f);

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }

        if (abs_error > tolerance) {
            error_count++;
        }
    }

    std::cout << "  最大绝对误差: " << max_abs_error << std::endl;
    std::cout << "  最大相对误差: " << max_rel_error << std::endl;
    std::cout << "  错误元素数: " << error_count << " / " << m * n << std::endl;

    bool passed = (error_count == 0);
    return passed;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    // 默认参数
    int m = 128;
    int n = 256;
    int k = 128;
    int nnz_per_row = 32;
    int k_version = 128;
    bool verbose = false;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--m" && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        } else if (arg == "--nnz_per_row" && i + 1 < argc) {
            nnz_per_row = std::atoi(argv[++i]);
        } else if (arg == "--k_version" && i + 1 < argc) {
            k_version = std::atoi(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --m M              Number of rows in sparse matrix A (default: 128)" << std::endl;
            std::cout << "  --n N              Number of columns in dense matrix B (default: 256)" << std::endl;
            std::cout << "  --k K              Number of columns in A / rows in B (default: 128)" << std::endl;
            std::cout << "  --nnz_per_row N    Average nonzeros per row (default: 32)" << std::endl;
            std::cout << "  --k_version V     Kernel version: 32 or 128 (default: 128)" << std::endl;
            std::cout << "  --verbose, -v      Verbose output" << std::endl;
            std::cout << "  --help, -h         Show this help message" << std::endl;
            return 0;
        }
    }

    // 验证参数
    if (k_version != 32 && k_version != 128) {
        std::cerr << "Error: k_version must be 32 or 128" << std::endl;
        return 1;
    }

    if (k != k_version) {
        std::cerr << "Warning: k=" << k << " doesn't match k_version=" << k_version << std::endl;
    }

    int seg_length = (k_version == 32) ? 512 : 32;
    int k_block = 32;
    int vec_len = 4;

    std::cout << "========================================" << std::endl;
    std::cout << "  RoDe SPMM 测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "参数:" << std::endl;
    std::cout << "  m (sparse rows): " << m << std::endl;
    std::cout << "  k (sparse cols / dense rows): " << k << std::endl;
    std::cout << "  n (dense cols): " << n << std::endl;
    std::cout << "  nnz_per_row: " << nnz_per_row << std::endl;
    std::cout << "  k_version: " << k_version << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> col_dist(0, k - 1);
    std::uniform_real_distribution<> val_dist(-1.0f, 1.0f);

    // 生成 CSR 格式的稀疏矩阵 A (m x k)
    std::vector<std::vector<std::pair<int, float>>> csr_rows(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < nnz_per_row; ++j) {
            int col = col_dist(gen);
            float val = val_dist(gen);
            csr_rows[i].emplace_back(col, val);
        }
        // 去重
        std::sort(csr_rows[i].begin(), csr_rows[i].end());
        auto last = std::unique(csr_rows[i].begin(), csr_rows[i].end(),
            [](const auto& a, const auto& b) { return a.first == b.first; });
        csr_rows[i].erase(last, csr_rows[i].end());
    }

    // 构建 CSR 数组
    int nnz = 0;
    for (const auto& row : csr_rows) {
        nnz += row.size();
    }

    std::vector<int> row_offsets(m + 1);
    std::vector<int> column_indices(nnz);
    std::vector<float> values(nnz);

    row_offsets[0] = 0;
    int idx = 0;
    for (int i = 0; i < m; ++i) {
        for (const auto& [col, val] : csr_rows[i]) {
            column_indices[idx] = col;
            values[idx] = val;
            ++idx;
        }
        row_offsets[i + 1] = idx;
    }

    // 生成稠密矩阵 B (k x n)
    std::vector<float> B(k * n);
    for (auto& val : B) {
        val = val_dist(gen);
    }

    std::cout << "数据生成完成:" << std::endl;
    std::cout << "  稀疏矩阵 A: " << m << " x " << k << ", nnz = " << nnz << std::endl;
    std::cout << "  稠密矩阵 B: " << k << " x " << n << std::endl;
    std::cout << "  输出矩阵 C: " << m << " x " << n << std::endl;
    std::cout << std::endl;

    // RoDe 预处理
    std::cout << "执行 RoDe 预处理..." << std::endl;
    RoDeMetadata meta = rode_preprocess(row_offsets.data(), nnz, m, seg_length, k_block, vec_len);
    std::cout << "  m1 (block部分): " << meta.m1 << std::endl;
    std::cout << "  m2 (residue部分): " << meta.m2 << std::endl;
    std::cout << std::endl;

    // GPU 内存分配
    int* d_row_offsets;
    int* d_column_indices;
    int* d_row_indices1;
    int* d_row_indices2;
    int* d_row_seg_st_offsets;
    float* d_values;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices1, meta.m1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_indices2, meta.m2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_seg_st_offsets, (meta.m1 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    // 复制数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_row_offsets, row_offsets.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, column_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_indices1, meta.block_indices.data(), meta.m1 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_indices2, meta.residue_indices.data(), meta.m2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_seg_st_offsets, meta.st_offsets.data(), (meta.m1 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    // 清零输出矩阵
    CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));

    // 执行 RoDe SPMM
    std::cout << "执行 RoDe SPMM..." << std::endl;

    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;

    if (meta.m1 > 0) {
        CUDA_CHECK(cudaStreamCreate(&stream1));
    }
    if (meta.m2 > 0) {
        CUDA_CHECK(cudaStreamCreate(&stream2));
    }

    auto start_gpu = std::chrono::high_resolution_clock::now();

    if (k_version == 32) {
        RoDeSpmm_n32(meta.m1, meta.m2, k, n,
                      d_values, d_column_indices, d_row_offsets,
                      d_row_indices1, d_row_indices2, d_row_seg_st_offsets,
                      d_B, d_C, stream1, stream2);
    } else {
        RoDeSpmm_n128(meta.m1, meta.m2, k, n,
                       d_values, d_column_indices, d_row_offsets,
                       d_row_indices1, d_row_indices2, d_row_seg_st_offsets,
                       d_B, d_C, stream1, stream2);
    }

    if (stream1) CUDA_CHECK(cudaStreamSynchronize(stream1));
    if (stream2) CUDA_CHECK(cudaStreamSynchronize(stream2));

    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    if (stream1) CUDA_CHECK(cudaStreamDestroy(stream1));
    if (stream2) CUDA_CHECK(cudaStreamDestroy(stream2));

    std::cout << "  GPU 耗时: " << gpu_time << " us (" << gpu_time / 1000.0 << " ms)" << std::endl;
    std::cout << std::endl;

    // 复制结果回 CPU
    std::vector<float> gpu_result(m * n);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 参考计算
    std::cout << "执行 CPU 参考计算..." << std::endl;
    std::vector<float> cpu_result(m * n);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_spmm_reference(row_offsets.data(), column_indices.data(), values.data(), B.data(), cpu_result.data(), m, k, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    std::cout << "  CPU 耗时: " << cpu_time << " us (" << cpu_time / 1000.0 << " ms)" << std::endl;
    std::cout << std::endl;

    // 比较结果
    std::cout << "比较结果..." << std::endl;
    bool passed = compare_results(gpu_result.data(), cpu_result.data(), m, n);

    // 计算加速比
    float speedup = static_cast<float>(cpu_time) / static_cast<float>(gpu_time);
    std::cout << "  加速比: " << speedup << "x" << std::endl;
    std::cout << std::endl;

    // 打印前几个元素（如果 verbose）
    if (verbose) {
        std::cout << "前 10 个结果 (前 5 行, 前 2 列):" << std::endl;
        for (int i = 0; i < std::min(5, m); ++i) {
            for (int j = 0; j < std::min(2, n); ++j) {
                int idx = i * n + j;
                std::cout << "  C[" << i << "][" << j << "] CPU=" << cpu_result[idx]
                          << " GPU=" << gpu_result[idx]
                          << " diff=" << std::abs(gpu_result[idx] - cpu_result[idx]) << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // 测试结果
    if (passed) {
        std::cout << "✓ 测试通过!" << std::endl;
    } else {
        std::cout << "✗ 测试失败!" << std::endl;
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_column_indices));
    CUDA_CHECK(cudaFree(d_row_indices1));
    CUDA_CHECK(cudaFree(d_row_indices2));
    CUDA_CHECK(cudaFree(d_row_seg_st_offsets));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}
