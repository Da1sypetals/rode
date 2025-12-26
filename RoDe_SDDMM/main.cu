/**
 * RoDe SDDMM 测试程序
 * 
 * 该程序测试 RoDe SDDMM 算子的正确性：
 * 1. 随机生成 CSR 稀疏矩阵和两个稠密矩阵
 * 2. 对 CSR 矩阵进行 RoDe 格式预处理
 * 3. 调用 RoDe SDDMM 算子
 * 4. 与 CPU 参考实现进行对比验证
 * 
 * SDDMM 操作: out = S ⊙ (A × B^T)
 * 其中 S 是稀疏矩阵，A 是左侧稠密矩阵，B 是右侧稠密矩阵
 * 
 * 编译命令 (Tesla T4 - sm_75):
 *   nvcc -O3 -arch=sm_75 -std=c++17 -o test_sddmm main.cu RoDeSddmm.cu -I.
 * 
 * 运行:
 *   ./test_sddmm [m] [n] [k] [nnz_per_row]
 *   例如: ./test_sddmm 1024 2048 128 64
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

#include "RoDeSddmm.h"

// ============================================================================
// CUDA 错误检查宏
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// CPU 参考实现
// ============================================================================
/**
 * @brief CPU 上的 SDDMM 参考实现
 * 
 * 计算 out[i] = values[i] * dot(lhs[row[i]], rhs[col[i]])
 * 其中 row[i] 和 col[i] 是稀疏矩阵中第 i 个非零元素的行列索引
 * 
 * @param m 稀疏矩阵行数
 * @param n 稀疏矩阵列数
 * @param k 隐藏维度
 * @param nnz 非零元素数量
 * @param row_offsets CSR 行指针
 * @param column_indices CSR 列索引
 * @param values 稀疏矩阵值
 * @param lhs_matrix 左侧稠密矩阵 (m x k)
 * @param rhs_matrix 右侧稠密矩阵 (n x k)
 * @param out 输出结果
 */
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
            
            // 计算 dot product: lhs[row, :] · rhs[col, :]
            float dot = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                dot += lhs_matrix[row * k + kk] * rhs_matrix[col * k + kk];
            }
            
            // 乘以稀疏矩阵的值
            out[idx] = values[idx] * dot;
        }
    }
}

// ============================================================================
// RoDe 预处理函数
// ============================================================================
/**
 * @brief 将标准 CSR 格式转换为 RoDe SDDMM 所需的格式
 * 
 * RoDe 将稀疏矩阵的每一行分割为多个 segment：
 * - Block 部分：非零元素数量 >= kBlockItemsX 的完整 segment
 * - Residue 部分：每行剩余的不完整部分
 * 
 * @param SegmentLength 每个 segment 的长度
 * @param vectorLen 向量加载的对齐长度
 * @param KBLOCK block 的大小
 * @param M 矩阵行数
 * @param row_ptr CSR 行指针
 * @param block_r_ind 输出：Block 部分行索引
 * @param st_off 输出：Block 部分起始偏移
 * @param residue_r_ind 输出：Residue 部分行索引
 * @param n_blk 输出：Block segment 数量
 * @param n_res 输出：Residue 部分数量
 */
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
        
        // 处理超过 SegmentLength 的部分（第一个 segment 需要考虑 padding）
        if (nnz > SegmentLength) {
            block_r_ind[n_blk] = i;
            st_off[n_blk++] = row_offset;
            row_offset = (row_offset + SegmentLength) - n_padding;
            nnz -= SegmentLength;
        }
        
        // 处理剩余的完整 segments
        while (nnz > SegmentLength) {
            block_r_ind[n_blk] = i;
            st_off[n_blk++] = row_offset;
            row_offset += SegmentLength;
            nnz -= SegmentLength;
        }
        
        // 处理最后的部分
        if (nnz > 0) {
            // 如果剩余元素 >= KBLOCK，添加到 block 部分
            if (nnz >= KBLOCK) {
                block_r_ind[n_blk] = i;
                st_off[n_blk++] = row_offset;
            }
            // 如果有不能被 KBLOCK 整除的残余，添加到 residue 部分
            if (nnz % KBLOCK) {
                residue_r_ind[n_res++] = i;
            }
        }
    }
    
    // st_off 需要多一个元素作为结束标记
    st_off[n_blk] = row_ptr[M];
}

// ============================================================================
// 随机生成 CSR 稀疏矩阵
// ============================================================================
/**
 * @brief 随机生成 CSR 格式的稀疏矩阵
 * 
 * @param m 行数
 * @param n 列数
 * @param nnz_per_row 每行平均非零元素数
 * @param row_offsets 输出：行偏移数组
 * @param column_indices 输出：列索引数组
 * @param values 输出：非零值数组
 * @param pad_to 行填充到的倍数（用于向量化加载）
 * @param seed 随机种子
 * @return 实际的非零元素数量
 */
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
    std::uniform_int_distribution<int> nnz_dist(1, nnz_per_row * 2);
    
    row_offsets.clear();
    column_indices.clear();
    values.clear();
    
    row_offsets.push_back(0);
    
    for (int i = 0; i < m; ++i) {
        // 随机确定这一行的非零元素数量
        int row_nnz = std::min(nnz_dist(gen), n);
        
        // 随机选择列索引
        std::vector<int> cols(n);
        std::iota(cols.begin(), cols.end(), 0);
        std::shuffle(cols.begin(), cols.end(), gen);
        cols.resize(row_nnz);
        std::sort(cols.begin(), cols.end());
        
        // 添加非零元素
        for (int col : cols) {
            column_indices.push_back(col);
            values.push_back(val_dist(gen));
        }
        
        // 行填充（可选）
        if (pad_to > 1) {
            int current_nnz = column_indices.size() - row_offsets.back();
            int residue = current_nnz % pad_to;
            if (residue > 0) {
                int to_add = pad_to - residue;
                for (int j = 0; j < to_add; ++j) {
                    // 填充零值，列索引使用最后一个有效列
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
// 结果验证
// ============================================================================
/**
 * @brief 比较 GPU 结果和 CPU 参考结果
 * 
 * @param gpu_result GPU 计算结果
 * @param cpu_result CPU 参考结果
 * @param n 元素数量
 * @param tolerance 容差
 * @return 是否通过验证
 */
bool verify_results(
    const float* gpu_result,
    const float* cpu_result,
    int n,
    float tolerance = 1e-3f)
{
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int error_count = 0;
    int first_error_idx = -1;
    
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        float rel_diff = diff / (std::abs(cpu_result[i]) + 1e-6f);
        
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
        
        if (diff > tolerance && rel_diff > tolerance) {
            if (first_error_idx < 0) {
                first_error_idx = i;
            }
            error_count++;
        }
    }
    
    std::cout << "=== 验证结果 ===" << std::endl;
    std::cout << "最大绝对误差: " << max_diff << std::endl;
    std::cout << "最大相对误差: " << max_rel_diff << std::endl;
    std::cout << "错误元素数量: " << error_count << " / " << n << std::endl;
    
    if (error_count > 0) {
        std::cout << "第一个错误位置: " << first_error_idx << std::endl;
        std::cout << "  GPU: " << gpu_result[first_error_idx] 
                  << ", CPU: " << cpu_result[first_error_idx] << std::endl;
    }
    
    return error_count == 0;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv)
{
    // 默认参数
    int m = 1024;       // 稀疏矩阵行数
    int n = 2048;       // 稀疏矩阵列数
    int k = 128;        // 隐藏维度 (32 或 128)
    int nnz_per_row = 64;  // 每行平均非零元素数
    
    // 解析命令行参数
    if (argc >= 2) m = atoi(argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);
    if (argc >= 5) nnz_per_row = atoi(argv[4]);
    
    // 验证参数
    if (k != 32 && k != 128) {
        std::cerr << "错误: k 必须是 32 或 128" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "RoDe SDDMM 测试" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "矩阵参数:" << std::endl;
    std::cout << "  稀疏矩阵大小: " << m << " x " << n << std::endl;
    std::cout << "  隐藏维度 k: " << k << std::endl;
    std::cout << "  每行平均非零元素: " << nnz_per_row << std::endl;
    std::cout << "============================================" << std::endl;
    
    // ========================================================================
    // 1. 生成测试数据
    // ========================================================================
    std::cout << "\n[1] 生成测试数据..." << std::endl;
    
    // 生成 CSR 稀疏矩阵
    std::vector<int> h_row_offsets, h_column_indices;
    std::vector<float> h_values;
    int nnz = generate_random_csr(m, n, nnz_per_row, 
                                   h_row_offsets, h_column_indices, h_values,
                                   4, 42);
    
    std::cout << "  实际非零元素数量 (含填充): " << nnz << std::endl;
    
    // 生成稠密矩阵
    std::vector<float> h_lhs_matrix, h_rhs_matrix;
    generate_random_dense(m, k, h_lhs_matrix, 123);
    generate_random_dense(n, k, h_rhs_matrix, 456);
    
    std::cout << "  左侧稠密矩阵大小: " << m << " x " << k << std::endl;
    std::cout << "  右侧稠密矩阵大小: " << n << " x " << k << std::endl;
    
    // ========================================================================
    // 2. CPU 参考计算
    // ========================================================================
    std::cout << "\n[2] CPU 参考计算..." << std::endl;
    
    std::vector<float> h_cpu_result(nnz, 0.0f);
    sddmm_cpu_reference(m, n, k, nnz,
                        h_row_offsets.data(),
                        h_column_indices.data(),
                        h_values.data(),
                        h_lhs_matrix.data(),
                        h_rhs_matrix.data(),
                        h_cpu_result.data());
    
    std::cout << "  CPU 计算完成" << std::endl;
    
    // ========================================================================
    // 3. RoDe 预处理
    // ========================================================================
    std::cout << "\n[3] RoDe 预处理..." << std::endl;
    
    // 根据 k 选择参数
    // k=32:  SEG_LENGTH=512, KBLOCK=32, vectorLen=4
    // k=128: SEG_LENGTH=32,  KBLOCK=32, vectorLen=4
    int SEG_LENGTH = (k == 32) ? 512 : 32;
    int KBLOCK = 32;
    int vectorLen = 4;
    
    std::cout << "  SEG_LENGTH: " << SEG_LENGTH << std::endl;
    std::cout << "  KBLOCK: " << KBLOCK << std::endl;
    
    // 预估最大 block 数量
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
    
    std::cout << "  Block segment 数量 (m1): " << n_blk << std::endl;
    std::cout << "  Residue 数量 (m2): " << n_res << std::endl;
    
    // ========================================================================
    // 4. 分配 GPU 内存并拷贝数据
    // ========================================================================
    std::cout << "\n[4] 分配 GPU 内存..." << std::endl;
    
    // CSR 数据
    int *d_row_offsets, *d_column_indices;
    float *d_values;
    CUDA_CHECK(cudaMalloc(&d_row_offsets, sizeof(int) * (m + 1)));
    CUDA_CHECK(cudaMalloc(&d_column_indices, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * nnz));
    
    // RoDe 预处理数据
    int *d_block_r_ind, *d_residue_r_ind, *d_st_off;
    CUDA_CHECK(cudaMalloc(&d_block_r_ind, sizeof(int) * std::max(n_blk, 1)));
    CUDA_CHECK(cudaMalloc(&d_residue_r_ind, sizeof(int) * std::max(n_res, 1)));
    CUDA_CHECK(cudaMalloc(&d_st_off, sizeof(int) * (n_blk + 1)));
    
    // 稠密矩阵
    float *d_lhs_matrix, *d_rhs_matrix;
    CUDA_CHECK(cudaMalloc(&d_lhs_matrix, sizeof(float) * m * k));
    CUDA_CHECK(cudaMalloc(&d_rhs_matrix, sizeof(float) * n * k));
    
    // 输出
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * nnz));
    
    std::cout << "  GPU 内存分配完成" << std::endl;
    
    // 拷贝数据到 GPU
    std::cout << "\n[5] 拷贝数据到 GPU..." << std::endl;
    
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
    
    // 初始化输出为 0
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * nnz));
    
    std::cout << "  数据拷贝完成" << std::endl;
    
    // ========================================================================
    // 6. 创建 CUDA 流并调用算子
    // ========================================================================
    std::cout << "\n[6] 执行 RoDe SDDMM 算子..." << std::endl;
    
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        if (k == 32) {
            RoDeSDDMM_n32(n_blk, n_res, n, k,
                          d_block_r_ind, d_residue_r_ind, d_st_off,
                          d_row_offsets, d_column_indices, d_values,
                          d_lhs_matrix, d_rhs_matrix, d_out,
                          stream1, stream2);
        } else {
            RoDeSDDMM_n128(n_blk, n_res, n, k,
                           d_block_r_ind, d_residue_r_ind, d_st_off,
                           d_row_offsets, d_column_indices, d_values,
                           d_lhs_matrix, d_rhs_matrix, d_out,
                           stream1, stream2);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int num_iterations = 100;
    
    CUDA_CHECK(cudaEventRecord(start, stream1));
    
    for (int i = 0; i < num_iterations; ++i) {
        if (k == 32) {
            RoDeSDDMM_n32(n_blk, n_res, n, k,
                          d_block_r_ind, d_residue_r_ind, d_st_off,
                          d_row_offsets, d_column_indices, d_values,
                          d_lhs_matrix, d_rhs_matrix, d_out,
                          stream1, stream2);
        } else {
            RoDeSDDMM_n128(n_blk, n_res, n, k,
                           d_block_r_ind, d_residue_r_ind, d_st_off,
                           d_row_offsets, d_column_indices, d_values,
                           d_lhs_matrix, d_rhs_matrix, d_out,
                           stream1, stream2);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    float avg_time_ms = elapsed_ms / num_iterations;
    std::cout << "  平均执行时间: " << avg_time_ms << " ms" << std::endl;
    
    // ========================================================================
    // 7. 拷贝结果回主机并验证
    // ========================================================================
    std::cout << "\n[7] 验证结果..." << std::endl;
    
    std::vector<float> h_gpu_result(nnz);
    CUDA_CHECK(cudaMemcpy(h_gpu_result.data(), d_out, 
                          sizeof(float) * nnz, cudaMemcpyDeviceToHost));
    
    bool passed = verify_results(h_gpu_result.data(), h_cpu_result.data(), nnz);
    
    std::cout << "\n============================================" << std::endl;
    if (passed) {
        std::cout << "✓ 测试通过！" << std::endl;
    } else {
        std::cout << "✗ 测试失败！" << std::endl;
    }
    std::cout << "============================================" << std::endl;
    
    // ========================================================================
    // 8. 清理资源
    // ========================================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_column_indices));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_block_r_ind));
    CUDA_CHECK(cudaFree(d_residue_r_ind));
    CUDA_CHECK(cudaFree(d_st_off));
    CUDA_CHECK(cudaFree(d_lhs_matrix));
    CUDA_CHECK(cudaFree(d_rhs_matrix));
    CUDA_CHECK(cudaFree(d_out));
    
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
