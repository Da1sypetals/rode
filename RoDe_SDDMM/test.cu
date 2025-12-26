/**
 * RoDe SDDMM 测试程序 (k=128)
 * 
 * 接收参数执行测试，输出 JSON 格式结果供 Python 脚本解析
 * 
 * 编译命令:
 *   nvcc -O3 -arch=sm_75 -std=c++17 -o test_k128 test.cu RoDeSddmm.cu -I.
 * 
 * 运行:
 *   ./test_k128 <m> <n> <nnz_per_row>
 * 
 * 输出 JSON 格式:
 *   {"passed": true/false, "mae": x, "mean_rel": x, "max_abs": x, "max_rel": x, "errors": x, "nnz": x}
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>

#include "RoDeSddmm.h"

// ============================================================================
// CUDA 错误检查宏
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "{\"passed\": false, \"error\": \"CUDA Error: " \
                  << cudaGetErrorString(err) << "\"}" << std::endl; \
        return EXIT_FAILURE; \
    } \
} while(0)

// ============================================================================
// CPU 参考实现
// ============================================================================
void sddmm_cpu_reference(
    int m, int n, int k, int nnz,
    const int* row_offsets,
    const int* column_indices,
    const float* values,
    const float* lhs_matrix,
    const float* rhs_matrix,
    float* out)
{
    for (int row = 0; row < m; ++row) {
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = column_indices[idx];
            
            float dot = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                dot += lhs_matrix[row * k + kk] * rhs_matrix[col * k + kk];
            }
            
            out[idx] = values[idx] * dot;
        }
    }
}

// ============================================================================
// RoDe 预处理函数
// ============================================================================
void RoDe_preprocess(
    int SegmentLength, int vectorLen, int KBLOCK, int M,
    const int* row_ptr,
    int* block_r_ind,
    int* st_off,
    int* residue_r_ind,
    int& n_blk,
    int& n_res)
{
    n_blk = 0;
    n_res = 0;
    
    for (int i = 0; i < M; ++i) {
        int row_offset = row_ptr[i];
        int n_padding = row_offset % vectorLen;
        int nnz = row_ptr[i + 1] - row_offset + n_padding;
        
        if (nnz > SegmentLength) {
            block_r_ind[n_blk] = i;
            st_off[n_blk++] = row_offset;
            row_offset = (row_offset + SegmentLength) - n_padding;
            nnz -= SegmentLength;
        }
        
        while (nnz > SegmentLength) {
            block_r_ind[n_blk] = i;
            st_off[n_blk++] = row_offset;
            row_offset += SegmentLength;
            nnz -= SegmentLength;
        }
        
        if (nnz > 0) {
            if (nnz >= KBLOCK) {
                block_r_ind[n_blk] = i;
                st_off[n_blk++] = row_offset;
            }
            if (nnz % KBLOCK) {
                residue_r_ind[n_res++] = i;
            }
        }
    }
    
    st_off[n_blk] = row_ptr[M];
}

// ============================================================================
// 随机生成 CSR 稀疏矩阵
// ============================================================================
int generate_random_csr(
    int m, int n, int nnz_per_row,
    std::vector<int>& row_offsets,
    std::vector<int>& column_indices,
    std::vector<float>& values,
    int pad_to = 4,
    unsigned int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::poisson_distribution<int> nnz_dist(nnz_per_row);
    
    row_offsets.clear();
    column_indices.clear();
    values.clear();
    
    row_offsets.push_back(0);
    
    for (int i = 0; i < m; ++i) {
        int row_nnz = std::max(1, std::min(nnz_dist(gen), n));
        
        std::vector<int> cols(n);
        std::iota(cols.begin(), cols.end(), 0);
        std::shuffle(cols.begin(), cols.end(), gen);
        cols.resize(row_nnz);
        std::sort(cols.begin(), cols.end());
        
        for (int col : cols) {
            column_indices.push_back(col);
            values.push_back(val_dist(gen));
        }
        
        if (pad_to > 1) {
            int current_nnz = column_indices.size() - row_offsets.back();
            int residue = current_nnz % pad_to;
            if (residue > 0) {
                int to_add = pad_to - residue;
                for (int j = 0; j < to_add; ++j) {
                    column_indices.push_back(cols.back());
                    values.push_back(0.0f);
                }
            }
        }
        
        row_offsets.push_back(column_indices.size());
    }
    
    return column_indices.size();
}

// ============================================================================
// 随机生成稠密矩阵
// ============================================================================
void generate_random_dense(
    int rows, int cols,
    std::vector<float>& matrix,
    unsigned int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

// ============================================================================
// 误差统计结构
// ============================================================================
struct ErrorStats {
    float mae;              // 平均绝对误差
    float mean_rel_error;   // 平均相对误差
    float max_abs_error;    // 最大绝对误差
    float max_rel_error;    // 最大相对误差
    int error_count;        // 超过阈值的错误数
    int total_count;        // 总元素数
    bool passed;            // 是否通过测试
};

// ============================================================================
// 结果验证
// ============================================================================
ErrorStats verify_results(
    const float* gpu_result,
    const float* cpu_result,
    int n,
    float tolerance = 1e-3f)
{
    ErrorStats stats;
    stats.mae = 0.0f;
    stats.mean_rel_error = 0.0f;
    stats.max_abs_error = 0.0f;
    stats.max_rel_error = 0.0f;
    stats.error_count = 0;
    stats.total_count = n;
    
    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;
    
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        float rel_diff = diff / (std::abs(cpu_result[i]) + 1e-6f);
        
        sum_abs_error += diff;
        sum_rel_error += rel_diff;
        
        if (diff > stats.max_abs_error) {
            stats.max_abs_error = diff;
        }
        if (rel_diff > stats.max_rel_error) {
            stats.max_rel_error = rel_diff;
        }
        
        if (diff > tolerance && rel_diff > tolerance) {
            stats.error_count++;
        }
    }
    
    stats.mae = static_cast<float>(sum_abs_error / n);
    stats.mean_rel_error = static_cast<float>(sum_rel_error / n);
    stats.passed = (stats.error_count == 0);
    
    return stats;
}

// ============================================================================
// 输出 JSON 结果
// ============================================================================
void output_json(const ErrorStats& stats) {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "{";
    std::cout << "\"passed\": " << (stats.passed ? "true" : "false") << ", ";
    std::cout << "\"mae\": " << stats.mae << ", ";
    std::cout << "\"mean_rel\": " << stats.mean_rel_error << ", ";
    std::cout << "\"max_abs\": " << stats.max_abs_error << ", ";
    std::cout << "\"max_rel\": " << stats.max_rel_error << ", ";
    std::cout << "\"errors\": " << stats.error_count << ", ";
    std::cout << "\"nnz\": " << stats.total_count;
    std::cout << "}" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv)
{
    // 参数检查
    if (argc != 4) {
        std::cerr << "{\"passed\": false, \"error\": \"Usage: " << argv[0] 
                  << " <m> <n> <nnz_per_row>\"}" << std::endl;
        return EXIT_FAILURE;
    }
    
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int nnz_per_row = atoi(argv[3]);
    const int k = 128;  // 固定 k=128
    
    // 参数验证
    if (m <= 0 || n <= 0 || nnz_per_row <= 0) {
        std::cerr << "{\"passed\": false, \"error\": \"Invalid parameters: m, n, nnz_per_row must be positive\"}" << std::endl;
        return EXIT_FAILURE;
    }
    
    // ========== 1. 生成测试数据 ==========
    std::vector<int> h_row_offsets, h_column_indices;
    std::vector<float> h_values;
    int nnz = generate_random_csr(m, n, nnz_per_row, 
                                   h_row_offsets, h_column_indices, h_values,
                                   4, 42);
    
    std::vector<float> h_lhs_matrix, h_rhs_matrix;
    generate_random_dense(m, k, h_lhs_matrix, 123);
    generate_random_dense(n, k, h_rhs_matrix, 456);
    
    // ========== 2. CPU 参考计算 ==========
    std::vector<float> h_cpu_result(nnz, 0.0f);
    sddmm_cpu_reference(m, n, k, nnz,
                        h_row_offsets.data(),
                        h_column_indices.data(),
                        h_values.data(),
                        h_lhs_matrix.data(),
                        h_rhs_matrix.data(),
                        h_cpu_result.data());
    
    // ========== 3. RoDe 预处理 ==========
    const int SEG_LENGTH = 32;
    const int KBLOCK = 32;
    const int vectorLen = 4;
    
    int max_blocks = nnz / SEG_LENGTH + m + 100;
    
    std::vector<int> h_block_r_ind(max_blocks);
    std::vector<int> h_st_off(max_blocks + 1);
    std::vector<int> h_residue_r_ind(m);
    int n_blk = 0, n_res = 0;
    
    RoDe_preprocess(SEG_LENGTH, vectorLen, KBLOCK, m,
                    h_row_offsets.data(),
                    h_block_r_ind.data(),
                    h_st_off.data(),
                    h_residue_r_ind.data(),
                    n_blk, n_res);
    
    // ========== 4. 分配 GPU 内存 ==========
    int *d_row_offsets, *d_column_indices;
    float *d_values;
    CUDA_CHECK(cudaMalloc(&d_row_offsets, sizeof(int) * (m + 1)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * nnz));
    
    int *d_block_r_ind, *d_residue_r_ind, *d_st_off;
    CUDA_CHECK(cudaMalloc(&d_block_r_ind, sizeof(int) * std::max(n_blk, 1)));
    CUDA_CHECK(cudaMalloc(&d_residue_r_ind, sizeof(int) * std::max(n_res, 1)));
    CUDA_CHECK(cudaMalloc(&d_st_off, sizeof(int) * (n_blk + 1)));
    
    float *d_lhs_matrix, *d_rhs_matrix;
    CUDA_CHECK(cudaMalloc(&d_lhs_matrix, sizeof(float) * m * k));
    CUDA_CHECK(cudaMalloc(&d_rhs_matrix, sizeof(float) * n * k));
    
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * nnz));
    
    // ========== 5. 拷贝数据 ==========
    CUDA_CHECK(cudaMemcpy(d_row_offsets, h_row_offsets.data(), 
                          sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, h_column_indices.data(), 
                          sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), 
                          sizeof(float) * nnz, cudaMemcpyHostToDevice));
    
    if (n_blk > 0) {
        CUDA_CHECK(cudaMemcpy(d_block_r_ind, h_block_r_ind.data(), 
                              sizeof(int) * n_blk, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_st_off, h_st_off.data(), 
                              sizeof(int) * (n_blk + 1), cudaMemcpyHostToDevice));
    }
    if (n_res > 0) {
        CUDA_CHECK(cudaMemcpy(d_residue_r_ind, h_residue_r_ind.data(), 
                              sizeof(int) * n_res, cudaMemcpyHostToDevice));
    }
    
    CUDA_CHECK(cudaMemcpy(d_lhs_matrix, h_lhs_matrix.data(), 
                          sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs_matrix, h_rhs_matrix.data(), 
                          sizeof(float) * n * k, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * nnz));
    
    // ========== 6. 执行算子 ==========
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    RoDeSDDMM_n128(n_blk, n_res, n, k,
                   d_block_r_ind, d_residue_r_ind, d_st_off,
                   d_row_offsets, d_column_indices, d_values,
                   d_lhs_matrix, d_rhs_matrix, d_out,
                   stream1, stream2);
    
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    // ========== 7. 验证结果 ==========
    std::vector<float> h_gpu_result(nnz);
    CUDA_CHECK(cudaMemcpy(h_gpu_result.data(), d_out, 
                          sizeof(float) * nnz, cudaMemcpyDeviceToHost));
    
    ErrorStats stats = verify_results(h_gpu_result.data(), h_cpu_result.data(), nnz);
    
    // ========== 8. 清理 ==========
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_values);
    cudaFree(d_block_r_ind);
    cudaFree(d_residue_r_ind);
    cudaFree(d_st_off);
    cudaFree(d_lhs_matrix);
    cudaFree(d_rhs_matrix);
    cudaFree(d_out);
    
    // ========== 9. 输出 JSON 结果 ==========
    output_json(stats);
    
    return stats.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
