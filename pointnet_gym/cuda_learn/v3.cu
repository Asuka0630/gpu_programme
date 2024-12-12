// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <dirent.h>
#include <cstring>
#include <unordered_map>
#include <hdf5/serial/H5Cpp.h>

// #define DEBUG 1
#define MAX_LB_POWER2(x) ((x) > 0 ? (1 << (31 - __builtin_clz(x))) : 0)
#define CUDA_CHECK(call)                                           \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }
#define CEIL(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0]) // a vector of 4 floats

std::unordered_map<std::string, std::vector<int>> weight_param_shape;
std::unordered_map<int, std::pair<std::string, int>> bn_layers;
cudaStream_t streams[15];

void set_weight_param_shape()
{
    weight_param_shape["feat.stn.conv1.weight.txt"] = {64, 3};
    weight_param_shape["feat.stn.conv2.weight.txt"] = {128, 64};
    weight_param_shape["feat.stn.conv3.weight.txt"] = {1024, 128};

    weight_param_shape["feat.stn.fc1.weight.txt"] = {512, 1024};
    weight_param_shape["feat.stn.fc2.weight.txt"] = {256, 512};
    weight_param_shape["feat.stn.fc3.weight.txt"] = {9, 256};

    weight_param_shape["feat.conv1.weight.txt"] = {64, 3};
    weight_param_shape["feat.fstn.conv1.weight.txt"] = {64, 64};
    weight_param_shape["feat.fstn.conv2.weight.txt"] = {128, 64};
    weight_param_shape["feat.fstn.conv3.weight.txt"] = {1024, 128};

    weight_param_shape["feat.fstn.fc1.weight.txt"] = {512, 1024};
    weight_param_shape["feat.fstn.fc2.weight.txt"] = {256, 512};
    weight_param_shape["feat.fstn.fc3.weight.txt"] = {4096, 256};

    weight_param_shape["feat.conv2.weight.txt"] = {128, 64};
    weight_param_shape["feat.conv3.weight.txt"] = {1024, 128};

    weight_param_shape["fc1.weight.txt"] = {512, 1024};
    weight_param_shape["fc2.weight.txt"] = {256, 512};
    weight_param_shape["fc3.weight.txt"] = {10, 256};
}
void set_bn_layers()
{
    bn_layers[0] = std::make_pair("bn2", 256);
    bn_layers[1] = std::make_pair("feat.stn.bn1", 64);
    bn_layers[2] = std::make_pair("feat.stn.bn2", 128);
    bn_layers[3] = std::make_pair("feat.stn.bn3", 1024);
    bn_layers[4] = std::make_pair("feat.stn.bn4", 512);
    bn_layers[5] = std::make_pair("feat.stn.bn5", 256);
    bn_layers[6] = std::make_pair("feat.bn1", 64);
    bn_layers[7] = std::make_pair("feat.fstn.bn1", 64);
    bn_layers[8] = std::make_pair("feat.fstn.bn2", 128);
    bn_layers[9] = std::make_pair("feat.fstn.bn3", 1024);
    bn_layers[10] = std::make_pair("feat.fstn.bn4", 512);
    bn_layers[11] = std::make_pair("feat.fstn.bn5", 256);
    bn_layers[12] = std::make_pair("feat.bn2", 128);
    bn_layers[13] = std::make_pair("feat.bn3", 1024);
    bn_layers[14] = std::make_pair("bn1", 512);
}
/**
 * @brief Tiled 2D matrix transpose
 * @attention no inplace
 * @param input: 输入数据, shape: [M, N]
 * @param output: 输出数据, shape: [N, M]
 * @param M
 * @param N
 */
template <typename T, int blockSize = 32>
__global__ void transpose2D(T *input, T *output, int M, int N)
{
    __shared__ T sdata[blockSize][blockSize + 1];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M)
    {
        sdata[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < M && y < N)
    {
        output[y * M + x] = sdata[threadIdx.x][threadIdx.y];
    }
}
template <typename T, int blockSize = 32>
__always_inline void transpose2D_call(T *input, T *output, int M, int N)
{
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL(N, blockSize), CEIL(M, blockSize));
    transpose2D<T, blockSize><<<gridDim, blockDim>>>(input, output, M, N);
}

/**
 * @brief Tiled matrix multiplication kernel AB+v
 * @attention v.shape=[K]
 * @param Md: M * N
 * @param Nd: N * K
 * @param Pd: M * K
 */
template <typename T, int TILED_SIZE>
__global__ void TiledGemmAddVec(T *Md, T *Nd, T *v, T *Pd, int M, int N, int K)
{
    __shared__ T Mds[TILED_SIZE][TILED_SIZE];
    __shared__ T Nds[TILED_SIZE][TILED_SIZE + 1];
    __shared__ T vs[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0)
        vs[threadIdx.x] = x >= K ? 0.0 : v[x];
    __syncthreads();
    T Pvalue = vs[threadIdx.x];
#pragma unroll 4
    for (int i = 0; i < N; i += TILED_SIZE)
    {
        if (i + threadIdx.x >= N || y >= M)
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Mds[threadIdx.y][threadIdx.x] = Md[y * N + i + threadIdx.x];
        if (i + threadIdx.y >= N || x >= K)
            Nds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Nds[threadIdx.y][threadIdx.x] = Nd[(i + threadIdx.y) * K + x];

        __syncthreads(); // 等待所有线程加载完毕
#pragma unroll
        for (int k = 0; k < TILED_SIZE; ++k)
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= K)
        return;
    Pd[y * K + x] = Pvalue;
}
template <typename T, int TILED_SIZE>
__global__ void Conv1d1k64ci_BN_ReLU(T *Md, T *Nd, T *v, T *Pd, T *bn_weight, T *bn_bias, int M, int N, int K)
{
    __shared__ T Mds[TILED_SIZE][TILED_SIZE];
    __shared__ T Nds[TILED_SIZE][TILED_SIZE + 1];
    __shared__ T vs[TILED_SIZE];
    __shared__ T alpha[TILED_SIZE];
    __shared__ T gamma[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0)
    {
        vs[threadIdx.x] = x >= K ? 0.0 : v[x];
        alpha[threadIdx.x] = x >= K ? 0.0 : bn_weight[x];
        gamma[threadIdx.x] = x >= K ? 0.0 : bn_bias[x];
    }
    __syncthreads();
    T Pvalue = vs[threadIdx.x];
#pragma unroll 4
    for (int i = 0; i < N; i += TILED_SIZE)
    {
        if (i + threadIdx.x >= N || y >= M)
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Mds[threadIdx.y][threadIdx.x] = Md[y * N + i + threadIdx.x];
        if (i + threadIdx.y >= N || x >= K)
            Nds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Nds[threadIdx.y][threadIdx.x] = Nd[(i + threadIdx.y) * K + x];

        __syncthreads(); // 等待所有线程加载完毕
#pragma unroll
        for (int k = 0; k < TILED_SIZE; ++k)
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= K)
        return;
    Pvalue = Pvalue * alpha[threadIdx.x] + gamma[threadIdx.x];
    Pd[y * K + x] = Pvalue > 0 ? Pvalue : 0.0;
}
template <typename T, int TILED_SIZE = 16, int TILED_SIZE_N = 3>
__global__ void Conv1d1k3ci_BN_ReLU(T *Md, T *Nd, T *v, T *Pd, T *bn_weight, T *bn_bias, int M, int N, int K)
{
    // Md: M * N
    // Nd: K * N
    __shared__ T Mds[TILED_SIZE][TILED_SIZE_N];
    __shared__ T Nds[TILED_SIZE_N][TILED_SIZE_N];
    __shared__ T vs[TILED_SIZE];
    __shared__ T alpha[TILED_SIZE];
    __shared__ T gamma[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0)
    {
        vs[threadIdx.x] = (x >= K ? 0.0 : v[x]);
        alpha[threadIdx.x] = (x >= K ? 0.0 : bn_weight[x]);
        gamma[threadIdx.x] = (x >= K ? 0.0 : bn_bias[x]);
    }
    __syncthreads();
    T Pvalue = vs[threadIdx.x];
#pragma unroll 4
    for (int i = 0; i < N; i += TILED_SIZE_N)
    {
        if (i + threadIdx.x < N || y < M)
            Mds[threadIdx.y][threadIdx.x] = Md[y * N + i + threadIdx.x];
        else
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        if (i + threadIdx.y < N && x < K && threadIdx.y < TILED_SIZE_N)
            Nds[threadIdx.y][threadIdx.x] = Nd[(i + threadIdx.y) * K + x];
        else if (threadIdx.y < TILED_SIZE_N)
            Nds[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads(); // 等待所有线程加载完毕
        for (int k = 0; k < TILED_SIZE_N; ++k)
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= K)
        return;
    Pvalue = Pvalue * alpha[threadIdx.x] + gamma[threadIdx.x];
    Pd[y * K + x] = Pvalue > 0 ? Pvalue : 0.0;
}
/**
 * @brief Tiled matrix multiplication kernel AB+v
 * @attention v.shape=[N]
 * @param a: M * N
 * @param b: N * K
 * @param c: M * K
 * @attention M>=128, N>=128, K>=8
 */
__global__ void Tiled128GemmAddVec(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, float *__restrict__ v, const int M, const int N, const int K)
{
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];
    __shared__ float s_v[BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid / (BK >> 2);
    int load_a_smem_k = (tid % (BK >> 2)) << 2;
    int load_b_smem_k = tid / (BN >> 2);
    int load_b_smem_n = (tid % (BN >> 2)) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    // 做一次加载
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        if (tid < (BN >> 2))
            FLOAT4(s_v[load_b_smem_n]) = FLOAT4(v[load_b_gmem_n]);
    }
    __syncthreads();
    int smem_sel_next;
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {
        // 加载第i个块
        int smem_sel = (bk - 1) & 1;
        smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
        // 计算第i-1个块
#pragma unroll 4
        for (int tk = 0; tk < BK; tk++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

#pragma unroll 4
    for (int tk = 0; tk < BK; tk++)
    {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
    }

#pragma unroll
    for (int tm = 0; tm < TM; tm++)
#pragma unroll
        for (int tn = 0; tn < TN; tn++)
            r_c[tm][tn] += s_v[(tx * TN / 2) + tn % 4 + (tn < 4 ? 0 : BN / 2)];

#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
__global__ void Conv1d1k128ci_BN(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, float *__restrict__ v, float *__restrict__ alpha, float *__restrict__ gamma, const int M, const int N, const int K)
{
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];
    __shared__ float s_v[3][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid / (BK >> 2);
    int load_a_smem_k = (tid % (BK >> 2)) << 2;
    int load_b_smem_k = tid / (BN >> 2);
    int load_b_smem_n = (tid % (BN >> 2)) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    // 做一次加载
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        if (tid < (BN >> 2))
        {
            FLOAT4(s_v[0][load_b_smem_n]) = FLOAT4(v[load_b_gmem_n]);
            FLOAT4(s_v[1][load_b_smem_n]) = FLOAT4(alpha[load_b_gmem_n]);
            FLOAT4(s_v[2][load_b_smem_n]) = FLOAT4(gamma[load_b_gmem_n]);
        }
    }
    __syncthreads();
    int smem_sel_next;
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {
        // 加载第i个块
        int smem_sel = (bk - 1) & 1;
        smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
        // 计算第i-1个块
#pragma unroll 4
        for (int tk = 0; tk < BK; tk++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

#pragma unroll 4
    for (int tk = 0; tk < BK; tk++)
    {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
    }

#pragma unroll
    for (int tm = 0; tm < TM; tm++)
#pragma unroll
        for (int tn = 0; tn < TN; tn++)
        {
            int col = (tx * TN / 2) + tn % 4 + (tn < 4 ? 0 : BN / 2);
            r_c[tm][tn] += s_v[0][col];
            r_c[tm][tn] = r_c[tm][tn] * s_v[1][col] + s_v[2][col];
        }

#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
__global__ void Conv1d1k128ci_BN_ReLU(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, float *__restrict__ v, float *__restrict__ alpha, float *__restrict__ gamma, const int M, const int N, const int K)
{
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];
    __shared__ float s_v[3][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid / (BK >> 2);
    int load_a_smem_k = (tid % (BK >> 2)) << 2;
    int load_b_smem_k = tid / (BN >> 2);
    int load_b_smem_n = (tid % (BN >> 2)) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    // 做一次加载
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        if (tid < (BN >> 2))
        {
            FLOAT4(s_v[0][load_b_smem_n]) = FLOAT4(v[load_b_gmem_n]);
            FLOAT4(s_v[1][load_b_smem_n]) = FLOAT4(alpha[load_b_gmem_n]);
            FLOAT4(s_v[2][load_b_smem_n]) = FLOAT4(gamma[load_b_gmem_n]);
        }
    }
    __syncthreads();
    int smem_sel_next;
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {
        // 加载第i个块
        int smem_sel = (bk - 1) & 1;
        smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
        // 计算第i-1个块
#pragma unroll 4
        for (int tk = 0; tk < BK; tk++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

#pragma unroll 4
    for (int tk = 0; tk < BK; tk++)
    {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
    }

#pragma unroll
    for (int tm = 0; tm < TM; tm++)
#pragma unroll
        for (int tn = 0; tn < TN; tn++)
        {
            int col = (tx * TN / 2) + tn % 4 + (tn < 4 ? 0 : BN / 2);
            r_c[tm][tn] += s_v[0][col];                            // fusion bias
            r_c[tm][tn] = r_c[tm][tn] * s_v[1][col] + s_v[2][col]; // fusion preprocessed bn_weight
            r_c[tm][tn] = r_c[tm][tn] > 0 ? r_c[tm][tn] : 0;       // fusion preprocessed bn_bias
        }

#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

/**
 * @brief AB^T + v + I
 * @attention v.shape = [K] Nd.shape = [K, N]
 * @param Md: M * N
 * @param Nd: K * N
 * @param Pd: M * K
 * @param M
 * @param N
 * @param K
 * @param r: sqrt(K)
 */
template <typename T, int TILED_SIZE = 32>
__global__ void TiledGemmAddVecAndBasis(T *Md, T *Nd, T *v, T *Pd, int M, int N, int K, int r)
{
    // Md: M * N
    // Nd: K * N
    __shared__ T Mds[TILED_SIZE][TILED_SIZE];
    __shared__ T Nds[TILED_SIZE][TILED_SIZE + 1];
    __shared__ T vs[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0)
        vs[threadIdx.x] = (x >= K ? 0.0 : (v[x] + (x % r == x / r ? 1.0 : 0.0)));
    __syncthreads();
    T Pvalue = vs[threadIdx.x];
#pragma unroll 4
    for (int i = 0; i < N; i += TILED_SIZE)
    {
        if (i + threadIdx.x >= N || y >= M)
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Mds[threadIdx.y][threadIdx.x] = Md[y * N + i + threadIdx.x];
        if (i + threadIdx.y >= N || x >= K)
            Nds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Nds[threadIdx.y][threadIdx.x] = Nd[(i + threadIdx.y) * K + x];

        __syncthreads(); // 等待所有线程加载完毕
        for (int k = 0; k < TILED_SIZE; ++k)
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= K)
        return;
    Pd[y * K + x] = Pvalue;
}
/**
 * @brief batch Gemm
 * @param Md: input,  [batch_size, M, N]
 * @param Nd: input,  [batch_size, N, K]
 * @param Pd: output, [batch_size, M, K]
 * @param batch_size
 * @param M
 * @param N
 * @param K
 */
template <typename T, int blockSize = 32>
__global__ void BatchGemm(T *Md, T *Nd, T *Pd, int batch_size, int M, int N, int K)
{
    __shared__ T Mds[blockSize][blockSize];
    __shared__ T Nds[blockSize][blockSize + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z; // blockDim.z = 1, threadIdx.z = 0
    int Md_batch_offset = z * M * N;
    int Nd_batch_offset = z * N * K;
    int Pd_batch_offset = z * M * K;
    T Pvalue = 0;
#pragma unroll 4
    for (int i = 0; i < N; i += blockSize)
    {
        if (i + threadIdx.x >= N || y >= M)
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Mds[threadIdx.y][threadIdx.x] = Md[Md_batch_offset + y * N + i + threadIdx.x];
        if (i + threadIdx.y >= N || x >= K)
            Nds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Nds[threadIdx.y][threadIdx.x] = Nd[Nd_batch_offset + (i + threadIdx.y) * K + x];

        __syncthreads(); // 等待所有线程加载完毕
#pragma unroll
        for (int k = 0; k < blockSize; ++k)
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= K)
        return;
    Pd[Pd_batch_offset + y * K + x] = Pvalue;
}

/**
 * @brief: 预先处理BN层的数据
 *
 */
template <typename T, int blockSize>
__global__ void PreprocessBatchNormParam(T *weight, T *bias, T *running_mean, T *running_var, int in_channel)
{
    // weight <- weight / sqrt(var+eps)
    // bias   <- bias - mean * weight / sqrt(var+eps)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= in_channel)
        return;
    T var = running_var[x], mean = running_mean[x];
    T w = weight[x], b = bias[x];
    T alpha = w * __frsqrt_rn(var + 1e-5);
    weight[x] = alpha;
    bias[x] = b - mean * alpha;
}
/**
 * @brief PreprocessedBatchNorm1d
 * @param input: 输入数据, shape: [batch_size, N, in_channel]
 * @param output: 输出数据, shape: [batch_size, N, in_channel]
 * @param weight: 权重, shape: [in_channel]
 * @param bias: 偏置, shape: [in_channel]
 */
template <typename T, int TILED_SIZE = 32>
__global__ void PreprocessedBatchNorm1d(T *input, T *output, T *weight, T *bias, int batch_size, int N, int in_channel)
{
    /*
        (input-mean)/sqrt(var+eps)*weight+bias
        = input * gamma + beta
        gamma = weight / sqrt(var+eps)
        beta = bias - mean * weight / sqrt(var+eps)

        input[i][j] = input[i][j] * gamma[j] + beta[j]
        gamma[j] = weight[j] / sqrt(var[j]+eps)
        beta[j] = bias[j] - mean[j] * weight[j] / sqrt(var[j]+eps)

        +--------+--------+
        |        |        |
        |  TILE  |  TILE  |
        |        |        |
        +--------+--------+
        |        |        |
        |  TILE  |  TILE  |
        |        |        |
        +--------+--------+

     */
    __shared__ T gamma[TILED_SIZE];
    __shared__ T beta[TILED_SIZE];
    int n = batch_size * N;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= in_channel || y >= n)
        return;
    if (threadIdx.y == 0)
    {
        gamma[threadIdx.x] = weight[x];
        beta[threadIdx.x] = bias[x];
    }
    __syncthreads();
    output[y * in_channel + x] = input[y * in_channel + x] * gamma[threadIdx.x] + beta[threadIdx.x];
}
template <typename T, int TILED_SIZE = 32>
__global__ void PreprocessedBatchNorm1dWithReLU(T *input, T *output, T *weight, T *bias, int batch_size, int N, int in_channel)
{
    __shared__ T gamma[TILED_SIZE];
    __shared__ T beta[TILED_SIZE];
    int n = batch_size * N;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= in_channel || y >= n)
        return;
    if (threadIdx.y == 0)
    {
        gamma[threadIdx.x] = weight[x];
        beta[threadIdx.x] = bias[x];
    }
    __syncthreads();
    T d = input[y * in_channel + x] * gamma[threadIdx.x] + beta[threadIdx.x];
    d = d < 0 ? 0 : d;
    output[y * in_channel + x] = d;
}

/**
 * @brief unroll warp
 * @param sdata: shared memory
 * @param tid: thread id
 */
template <typename T, int blockSize>
__device__ void WarpMaxReduce(volatile T *sdata, int tid)
{
    if (blockSize >= 64)
        sdata[tid] = fmax(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32)
        sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16)
        sdata[tid] = fmax(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)
        sdata[tid] = fmax(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4)
        sdata[tid] = fmax(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2)
        sdata[tid] = fmax(sdata[tid], sdata[tid + 1]);
}

/**
 * @brief exp sum_reduce
 * @param g_idata: shared memory, shape: [blockSize]
 * @param tid: thread.x
 */
template <typename T, int blockSize>
__device__ void WarpSumReduce(volatile T *sdata, int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

/**
 * @brief max reduce
 * @param g_idata: 输入数据, shape: [B, N, in_channel]
 * @param g_odata: 输出数据, shape: [B, blks, in_channel]
 * @param N
 */
template <typename T, int blockSize = 64>
__global__ void _maxReduce(T *g_idata, T *g_odata, int N)
{
    /*
        blockDim.x = 1, blockDim.y = blockSize, blockDim.z = 1
        gridDim.x = C, gridDim.y = 2^[log2(N)]/(blockSize), gridDim.z = B

        For example:
            g_idata = [B, 1157, C]
            blockSize = 256
            1157->1024+133
            =>
                block = {z: 1, y: 256, x: 1}
                grid = {z: B, y: 4, x: C}
    */
    extern __shared__ T sdata[];
    int tid = threadIdx.y;
    int g_idata_planeSize = N * gridDim.x;
    int g_idata_batch_base = g_idata_planeSize * blockIdx.z;
    int line_stride = blockDim.y * gridDim.y;
    int stride = line_stride * gridDim.x;

    int i = g_idata_batch_base + (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x;
    sdata[tid] = g_idata[i]; // 或 -T_MIN
    int cnt = blockIdx.y * blockDim.y + tid + line_stride;
    i += stride;
    while (cnt < N) // 将不足构成下一个block的部分进行算
    {
        sdata[tid] = fmax(sdata[tid], g_idata[i]);
        cnt += line_stride;
        i += stride;
    }
    __syncthreads();
    // for (int s = blockDim.y >> 1; s > 32; s >>= 1)
    // {
    //     if (tid < s)
    //         sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    //     __syncthreads();
    // }
    if (blockSize >= 512)
    {
        if (tid < 256)
            sdata[tid] = fmax(sdata[tid], sdata[tid + 256]);
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            sdata[tid] = fmax(sdata[tid], sdata[tid + 128]);
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
            sdata[tid] = fmax(sdata[tid], sdata[tid + 64]);
        __syncthreads();
    }
    // last warp unroll
    if (tid < 32)
    {
        if (blockSize > 2)
            WarpMaxReduce<T, blockSize / 2>(sdata, tid);
        else
            sdata[0] = fmax(sdata[0], sdata[1]);
    }
    int g_odata_batch_base = blockIdx.z * gridDim.x * gridDim.y;
    if (tid == 0)
        g_odata[g_odata_batch_base + blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
}

/**
 * @brief log_softmax
 * @param input shape=[Batch_size, dim]
 * @param output shape=[Batch_size, dim]
 * @param
 */
template <typename T>
__global__ void LogSoftmax_10(T *input, T *output, int batch_size, int dim)
{
    __shared__ T sdata[32];
    int tid = threadIdx.x;
    int batch_offset = blockIdx.z * dim;
    sdata[tid] = input[batch_offset + tid];
    __syncthreads();
    sdata[tid + 10] = sdata[tid]; // backup
    // last warp unroll
    if (tid < 4)
        WarpMaxReduce<T, 8>(sdata, tid);
    // __syncthreads();
    T max_val = fmax(sdata[0], fmax(sdata[8], sdata[9]));
    sdata[tid + 10] = sdata[tid + 10] - max_val;
    sdata[tid + 2 * 10] = expf(sdata[tid + 10]);
    // last warp unroll
    // __syncthreads();
    if (tid < 4)
        WarpSumReduce<T, 8>(sdata + 2 * 10, tid);
    // __syncthreads();
    T sum = sdata[2 * 10] + sdata[2 * 10 + 8] + sdata[2 * 10 + 9];
    output[batch_offset + tid] = sdata[tid + 10] - logf(sum);
}

/**
 * @brief argmax
 * @attention blockSize >= dim
 * @attention no inplace
 * @param input: shape=[N, dim]
 * @param output: shape=[N]
 */
template <typename T, int blockSize = 16>
__global__ void argmax(T *input, int *output, int N, int dim)
{
    int tid = threadIdx.x;
    int bid = blockIdx.y;
    __shared__ T shm_val[blockSize];
    __shared__ int shm_idx[blockSize];
    shm_val[tid] = tid < dim ? input[bid * dim + tid] : -1e9;
    shm_idx[tid] = tid;
    __syncthreads();
    T tm;
    if (tid < blockSize / 2)
    {
        tm = fmax(shm_val[tid], shm_val[tid + 8]);
        shm_idx[tid] = tm == shm_val[tid + 8] ? shm_idx[tid + 8] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = fmax(shm_val[tid], shm_val[tid + 4]);
        shm_idx[tid] = tm == shm_val[tid + 4] ? shm_idx[tid + 4] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = fmax(shm_val[tid], shm_val[tid + 2]);
        shm_idx[tid] = tm == shm_val[tid + 2] ? shm_idx[tid + 2] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = fmax(shm_val[tid], shm_val[tid + 1]);
        shm_idx[tid] = tm == shm_val[tid + 1] ? shm_idx[tid + 1] : shm_idx[tid];
    }
    output[bid] = shm_idx[0];
}
template <typename T, int blockSize = 16>
__global__ void argmaxWitLabels(T *input, int *output, int *labels, int N, int dim)
{
    int tid = threadIdx.x;
    int bid = blockIdx.y;
    __shared__ T shm_val[blockSize];
    __shared__ int shm_idx[blockSize];
    shm_val[tid] = tid < dim ? input[bid * dim + tid] : -1e9;
    shm_idx[tid] = tid;
    __syncthreads();
    T tm;
    if (tid < blockSize / 2)
    {
        tm = fmax(shm_val[tid], shm_val[tid + 8]);
        shm_idx[tid] = tm == shm_val[tid + 8] ? shm_idx[tid + 8] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = fmax(shm_val[tid], shm_val[tid + 4]);
        shm_idx[tid] = tm == shm_val[tid + 4] ? shm_idx[tid + 4] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = fmax(shm_val[tid], shm_val[tid + 2]);
        shm_idx[tid] = tm == shm_val[tid + 2] ? shm_idx[tid + 2] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = fmax(shm_val[tid], shm_val[tid + 1]);
        shm_idx[tid] = tm == shm_val[tid + 1] ? shm_idx[tid + 1] : shm_idx[tid];
    }
    output[bid] = (shm_idx[0] == labels[bid] ? 1 : 0);
}
/**
 * @brief sum reduce
 * @param g_idata: 输入数据, shape: [N]
 * @param g_odata: 输出数据, shape: [1]
 */
template <typename T, int blockSize>
__global__ void sumReduce(T *g_idata, T *g_odata, int n)
{
    // extern
    __shared__ T shm_data[blockSize];
    int tid = threadIdx.x;
    int i = tid;
    int gridSize = blockSize * 2 * gridDim.x;
    shm_data[tid] = 0;
    while (i < n)
    {
        shm_data[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
            shm_data[tid] += shm_data[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            shm_data[tid] += shm_data[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
            shm_data[tid] += shm_data[tid + 64];
        __syncthreads();
    }
    if (tid < 32)
        WarpSumReduce<T, blockSize>(shm_data, tid);
    if (tid == 0)
        atomicAdd(g_odata, shm_data[0]);
}

/**
 * @brief BatchGemm warpper
 * @attention 仅支持1x1卷积
 * @param Md: shape=[B,M,N]
 * @param Nd: shape=[B,N,K]
 * @param Pd: 输出数据指针
 * @param batch_size:
 * @param M:
 * @param N:
 * @param K:
 */
template <typename T, int TILED_SIZE = 32>
__always_inline void BatchGemm_call(T *Md, T *Nd, T *Pd, int batch_size, int M, int N, int K)
{
    dim3 blockDim(TILED_SIZE, TILED_SIZE, 1);
    dim3 gridDim(CEIL(K, blockDim.y), CEIL(M, blockDim.x), batch_size);
    BatchGemm<T, TILED_SIZE><<<gridDim, blockDim>>>(Md, Nd, Pd, batch_size, M, N, K);
}

/**
 * @brief 两级调用_maxReduce实现对[B, N, C]在C维度上求max
 * @attention tmp的大小需要大于[B, 2^[log2(N)] / (blockSize), C]
 * @attention blockSize必须为2的幂
 * @param input: shape [B, N, C]
 * @param output: shape [B, C]
 */
template <typename T, int blockSize = 128>
void maxReduce_call(T *input, T *tmp, T *output, int batch_size, int N, int C)
{
    dim3 blockDim(1, blockSize, 1);
    int blks = CEIL(MAX_LB_POWER2(N), blockSize);
#ifdef DEBUG
    if (blks > 1024)
        std::cout
            << "Too big N or too small blockSize" << std::endl;
#endif
    dim3 gridDim(C, blks, batch_size);
    if (blks > 1)
        _maxReduce<T, blockSize><<<gridDim, blockDim, blockSize * sizeof(T)>>>(input, tmp, N);
    else
    {
        // no up level reduce is needed
        _maxReduce<T, blockSize><<<gridDim, blockDim, blockSize * sizeof(T)>>>(input, output, N);
        return;
    }
    // TODO: check whether synchronization is required
    // recurse to N = blks
    dim3 blockDim_(1, blks, 1);
    dim3 gridDim_(C, 1, batch_size);
    switch (blockDim_.y)
    {
    case 1024:
        _maxReduce<T, 1024><<<gridDim_, blockDim_, 1024 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 512:
        _maxReduce<T, 512><<<gridDim_, blockDim_, 512 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 256:
        _maxReduce<T, 256><<<gridDim_, blockDim_, 256 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 128:
        _maxReduce<T, 128><<<gridDim_, blockDim_, 128 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 64:
        _maxReduce<T, 64><<<gridDim_, blockDim_, 64 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 32:
        _maxReduce<T, 32><<<gridDim_, blockDim_, 32 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 16:
        _maxReduce<T, 16><<<gridDim_, blockDim_, 16 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 8:
        _maxReduce<T, 8><<<gridDim_, blockDim_, 8 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 4:
        _maxReduce<T, 4><<<gridDim_, blockDim_, 4 * sizeof(T)>>>(tmp, output, blks);
        break;
    case 2:
        _maxReduce<T, 2><<<gridDim_, blockDim_, 2 * sizeof(T)>>>(tmp, output, blks);
        break;
    default:
        return;
    }
}

/**
 * @brief 线性层
 * @param input shape=[N(B,n), in_channel]
 * @param weight shape=[in_channel, out_channel]
 * @param bias shape=
 * @param in_channel
 * @param out_channel
 */
template <typename T, int blockSize = 32>
__always_inline void Linear_call(T *input, T *weight, T *bias, T *output, int batch_size, int N, int in_channel, int out_channel)
{
    N = batch_size * N;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL(out_channel, blockDim.x), CEIL(N, blockDim.y));
    TiledGemmAddVec<T, blockSize><<<gridDim, blockDim>>>(input, weight, bias, output, N, in_channel, out_channel);
}
template <typename T>
__always_inline void LinearTile128_call(T *input, T *weight, T *bias, T *output, int batch_size, int N, int in_channel, int out_channel)
{
    N = batch_size * N;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(CEIL(out_channel, BN), CEIL(N, BM));
    Tiled128GemmAddVec<<<gridDim, blockDim>>>(input, weight, output, bias, N, out_channel, in_channel);
}

template <typename T, int blockSize = 32>
__always_inline void LinearAddCoordBasis_call(T *input, T *weight, T *bias, T *output, int batch_size, int N, int in_channel, int out_channel, int basis_dim)
{
    N = batch_size * N;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL(out_channel, blockDim.x), CEIL(N, blockDim.y));
    TiledGemmAddVecAndBasis<T, blockSize><<<gridDim, blockDim>>>(input, weight, bias, output, N, in_channel, out_channel, basis_dim);
}

/**
 * @brief conv1d-nb-relu-block
 * @param input shape=[batch_size, n, ci]
 * @param lin_weight shape=[co, ci]
 * @param lin_bias shape=[co]
 * @param runnning_mean shape=[co]
 * @param running_var shape=[co]
 * @param bn_weight shape=[co]
 * @param bn_bias shape=[co]
 * @param output shape=[batch_size, n, co]
 * @param batch_size
 * @param n
 * @param ci
 * @param co
 */
__always_inline void conv1dTile128_bn_relu_block_call(float *input, float *lin_weight, float *lin_bias, float *bn_weight, float *bn_bias, float *output, int batch_size, int n, int ci, int co)
{
    int M = batch_size * n;
    int N = co;
    int K = ci;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(CEIL(N, BN), CEIL(M, BM));
    Conv1d1k128ci_BN_ReLU<<<gridDim, blockDim>>>(input, lin_weight, output, lin_bias, bn_weight, bn_bias, M, N, K);
}
template <int blockSize = 16>
__always_inline void conv1dTile128_bn_block_call(float *input, float *lin_weight, float *lin_bias, float *bn_weight, float *bn_bias, float *output, int batch_size, int n, int ci, int co)
{
    int M = batch_size * n;
    int N = co;
    int K = ci;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(CEIL(N, BN), CEIL(M, BM));
    Conv1d1k128ci_BN<<<gridDim, blockDim>>>(input, lin_weight, output, lin_bias, bn_weight, bn_bias, M, N, K);
}
template <typename T, int blockSize = 32>
__always_inline void conv1d_bn_relu_block_call(T *input, T *lin_weight, T *lin_bias, T *bn_weight, T *bn_bias, T *output, int batch_size, int n, int ci, int co)
{
    if (n * batch_size >= 128 && ci >= 64 && co >= 128)
    {
        conv1dTile128_bn_relu_block_call(input, lin_weight, lin_bias, bn_weight, bn_bias, output, batch_size, n, ci, co);
        return;
    }
    int M = batch_size * n;
    int N = ci;
    int K = co;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL(K, blockDim.x), CEIL(M, blockDim.y));
    Conv1d1k64ci_BN_ReLU<T, blockSize><<<gridDim, blockDim>>>(input, lin_weight, lin_bias, output, bn_weight, bn_bias, M, N, K);
}
template <typename T, int blockSize = 16, int blockSize_ci = 3>
__always_inline void rectconv1d_bn_relu_block_call(T *input, T *lin_weight, T *lin_bias, T *bn_weight, T *bn_bias, T *output, int batch_size, int n, int ci, int co)
{
    int M = batch_size * n;
    int N = ci;
    int K = co;
    dim3 blockDim(blockSize_ci, blockSize);
    dim3 gridDim(CEIL(K, blockDim.x), CEIL(M, blockDim.y));
    Conv1d1k3ci_BN_ReLU<T, blockSize, blockSize_ci><<<gridDim, blockDim>>>(input, lin_weight, lin_bias, output, bn_weight, bn_bias, M, N, K);
}
template <int blockSize = 32>
__always_inline void mlp_bn_relu_block_f32call(float *input, float *lin_weight, float *lin_bias, float *bn_weight, float *bn_bias, float *output, int batch_size, int n, int ci, int co)
{
    // if (n * batch_size >= 128 && ci >= 64 && co >= 128)
    //     LinearTile128_call<float>(input, lin_weight, lin_bias, output, batch_size, n, ci, co);
    // else
    Linear_call<float, blockSize>(input, lin_weight, lin_bias, output, batch_size, n, ci, co);
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL(co, blockDim.x), CEIL(batch_size * n, blockDim.y));
    PreprocessedBatchNorm1dWithReLU<float, blockSize><<<gridDim, blockDim>>>(output, output, bn_weight, bn_bias, batch_size, n, co);
}

template <typename T, int blockSize = 8>
__always_inline void LogSoftmax_10_call(T *input, T *output, int batch_size, int dim)
{
    dim3 blockDim(dim, 1, 1);
    dim3 gridDim(1, 1, batch_size);
    LogSoftmax_10<float><<<gridDim, blockDim>>>(input, output, batch_size, dim);
}

/**
 * @brief argmax kernel caller
 * @attention no inplace
 * @param input shape=[batch_size, dim]
 * @param output shape=[batch_size]
 */
template <typename T>
__always_inline void argmaxWithLabels_call(T *input, int *output, int *labels, int N, int dim)
{
    dim3 blockDim(16, 1);
    dim3 gridDim(1, N);
    argmaxWitLabels<T, 16><<<gridDim, blockDim>>>(input, output, labels, N, dim);
}

/**
 * @brief sum reduce call
 * @attention blockSize必须为2的幂, 且大于等于n
 * @param input: 输入数据, shape: [N]
 * @param output: 输出数据, shape: [1]
 */
template <typename T>
__always_inline void sumReduce_call(T *input, T *output, int n)
{
    dim3 blockDim(512, 1);
    dim3 gridDim(1, 1);
    sumReduce<T, 512><<<gridDim, blockDim>>>(input, output, n);
}

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string &dir)
{
    std::vector<std::string> files;
    DIR *dp;
    struct dirent *entry;
    if ((dp = opendir(dir.c_str())) != NULL)
    {
        while ((entry = readdir(dp)) != NULL)
        {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos)
            {
                files.push_back(filename);
            }
        }
        closedir(dp);
    }
    else
        perror("opendir");
    return files;
}

std::string getFileNameFromPath(const std::string &path)
{
    size_t lastSlashPos = path.find_last_of("/");
    if (lastSlashPos == std::string::npos)
        return path;
    return path.substr(lastSlashPos + 1);
}
float *read_param(const std::string &filepath)
{
    // 读取 .txt 文件并转换为 std::vector<float>
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open())
    {
        float value;
        while (file >> value)
        {
            data.push_back(value);
        }
        file.close();
    }
    else
        std::cerr << "Unable to open file: " << filepath << std::endl;
    float *tmp, *ptr;
    CUDA_CHECK(cudaMalloc(&tmp, data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(tmp, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    // transpose all fc/conv.weight
    std::string fname = getFileNameFromPath(filepath);
    if (weight_param_shape.count(fname))
    {
        int size = weight_param_shape[fname][0] * weight_param_shape[fname][1];
        if (size != data.size())
            perror("size not match");
        CUDA_CHECK(cudaMalloc(&ptr, data.size() * sizeof(float)));
        transpose2D_call(tmp, ptr, weight_param_shape[fname][0], weight_param_shape[fname][1]);
        CUDA_CHECK(cudaFree(tmp));
    }
    else
        ptr = tmp;
    return ptr;
}

template <int blockSize = 32>
void preProcessBNLayer(std::unordered_map<std::string, float *> params)
{
    set_bn_layers();
    for (int i = 0; i < 15; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        float *running_mean = params[bn_layers[i].first + ".running_mean"];
        float *running_var = params[bn_layers[i].first + ".running_var"];
        float *weight = params[bn_layers[i].first + ".weight"];
        float *bias = params[bn_layers[i].first + ".bias"];
        PreprocessBatchNormParam<float, blockSize><<<CEIL(bn_layers[i].second, blockSize), blockSize>>>(weight, bias, running_mean, running_var, bn_layers[i].second);
    }

    for (int i = 0; i < 15; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
}

std::unordered_map<std::string, float *> read_params(std::string dir)
{
    // std::string dir = "."; // 当前目录
    std::unordered_map<std::string, float *> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto &file : param_files)
    {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        if (filename.find("num_batches_tracked") == std::string::npos)
            params[filename] = read_param(dir + "/" + file);
    }
    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }
    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/
void pre_process_points(std::vector<float> &x, hsize_t length)
{
    x.size() > length ? x.resize(length) : x.resize(length, 0.0f);
}

using namespace H5;
int read_h5_file(const std::string &file_path, float *list_of_points, int *list_of_labels, int fix_length)
{
    int cnt = 0;
    if (fix_length <= 0)
    {
        std::cerr << "fix_length must be greater than 0" << std::endl;
        return 0;
    }
    try
    {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);
        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++)
            dataset_names.push_back(file.getObjnameByIdx(i));
        // 读取每个数据集
        for (const auto &name : dataset_names)
        {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);
            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);
            // 定长处理
            pre_process_points(points, fix_length * dims[1]);
            // 存储点云数据
            // list_of_points.push_back(points);
            memcpy(list_of_points + cnt * fix_length * 3, points.data(), fix_length * 3 * sizeof(float));
            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            // list_of_labels.push_back(label);
            list_of_labels[cnt++] = label;
        }
    }
    catch (FileIException &error)
    {
        error.printErrorStack();
    }
    catch (DataSetIException &error)
    {
        error.printErrorStack();
    }
    catch (DataSpaceIException &error)
    {
        error.printErrorStack();
    }
    catch (DataTypeIException &error)
    {
        error.printErrorStack();
    }
    return cnt;
}

template <typename T>
void batch_infer(std::unordered_map<std::string, T *> &params, T *batch_data, T *blk_out[], T *tmp, T *batch_out, int truncated_length, int batch_size)
{
    // 推理batch
    T *blk1_out = blk_out[1], *blk2_out = blk_out[2], *blk3_out = blk_out[3], *blk4_out = blk_out[4], *blk5_out = blk_out[5], *blk6_out = blk_out[6], *blk7_out = blk_out[7], *blk8_out = blk_out[8], *blk9_out = blk_out[9], *blk10_out = blk_out[10], *blk11_out = blk_out[11], *blk12_out = blk_out[12];

    // stn
    rectconv1d_bn_relu_block_call<float, 32>(batch_data, params["feat.stn.conv1.weight"], params["feat.stn.conv1.bias"], params["feat.stn.bn1.weight"], params["feat.stn.bn1.bias"], blk1_out, batch_size, truncated_length, 3, 64);
    conv1d_bn_relu_block_call<float, 32>(blk1_out, params["feat.stn.conv2.weight"], params["feat.stn.conv2.bias"], params["feat.stn.bn2.weight"], params["feat.stn.bn2.bias"], blk2_out, batch_size, truncated_length, 64, 128);   // feat.stn.conv2.weight=[128, 64], feat.stn.conv2.bias=[128], feat.stn.bn2.running_mean=[128], feat.stn.bn2.running_var=[128], feat.stn.bn2.weight=[128], feat.stn.bn2.bias=[128]
    conv1d_bn_relu_block_call<float, 32>(blk2_out, params["feat.stn.conv3.weight"], params["feat.stn.conv3.bias"], params["feat.stn.bn3.weight"], params["feat.stn.bn3.bias"], blk3_out, batch_size, truncated_length, 128, 1024); // feat.stn.conv3.weight=[1024, 128], feat.stn.conv3.bias=[1024], feat.stn.bn3.running_mean=[1024], feat.stn.bn3.running_var=[1024], feat.stn.bn3.weight=[1024], feat.stn.bn3.bias=[1024]

    maxReduce_call<float, 128>(blk3_out, tmp, blk4_out, batch_size, truncated_length, 1024);

    mlp_bn_relu_block_f32call<32>(blk4_out, params["feat.stn.fc1.weight"], params["feat.stn.fc1.bias"], params["feat.stn.bn4.weight"], params["feat.stn.bn4.bias"], blk5_out, batch_size, 1, 1024, 512); // feat.stn.fc1.weight=[512, 1024], feat.stn.fc1.bias=[512], feat.stn.bn4.running_mean=[512], feat.stn.bn4.running_var=[512], feat.stn.bn4.weight=[512], feat.stn.bn4.bias=[512]
    mlp_bn_relu_block_f32call<32>(blk5_out, params["feat.stn.fc2.weight"], params["feat.stn.fc2.bias"], params["feat.stn.bn5.weight"], params["feat.stn.bn5.bias"], blk6_out, batch_size, 1, 512, 256);  // feat.stn.fc2.weight=[256, 512], feat.stn.fc2.bias=[256], feat.stn.bn5.running_mean=[256], feat.stn.bn5.running_var=[256], feat.stn.bn5.weight=[256], feat.stn.bn5.bias=[256]

    LinearAddCoordBasis_call<float, 16>(blk6_out, params["feat.stn.fc3.weight"], params["feat.stn.fc3.bias"], blk7_out, batch_size, 1, 256, 9, 3); // feat.stn.fc3.weight=[9, 256], feat.stn.fc3.bias=[9]

    // feat
    BatchGemm_call<float, 8>(batch_data, blk7_out, blk8_out, batch_size, truncated_length, 3, 3); // [B, N, 3] x [B, (3, 3)]

    rectconv1d_bn_relu_block_call<float, 32>(blk8_out, params["feat.conv1.weight"], params["feat.conv1.bias"], params["feat.bn1.weight"], params["feat.bn1.bias"], blk9_out, batch_size, truncated_length, 3, 64); // feat.conv1.weight=[64, 3], feat.conv1.bias=[64], feat.bn1.running_mean=[64], feat.bn1.running_var=[64], feat.bn1.weight=[64], feat.bn1.bias=[64]

    // fstn
    conv1d_bn_relu_block_call<float, 32>(blk9_out, params["feat.fstn.conv1.weight"], params["feat.fstn.conv1.bias"], params["feat.fstn.bn1.weight"], params["feat.fstn.bn1.bias"], blk1_out, batch_size, truncated_length, 64, 64);    // feat.fstn.conv1.weight=[64, 64], feat.fstn.conv1.bias=[64], feat.fstn.conv1.running_mean=[64], feat.fstn.conv1.running_var=[64], feat.fstn.conv1.weight=[64], feat.fstn.conv1.bias=[64]
    conv1d_bn_relu_block_call<float, 32>(blk1_out, params["feat.fstn.conv2.weight"], params["feat.fstn.conv2.bias"], params["feat.fstn.bn2.weight"], params["feat.fstn.bn2.bias"], blk2_out, batch_size, truncated_length, 64, 128);   // feat.fstn.conv2.weight=[64, 64], feat.fstn.conv2.bias=[64], feat.fstn.bn2.running_mean=[64], feat.fstn.bn2.running_var=[64], feat.fstn.bn2.weight=[64], feat.fstn.bn2.bias=[64]
    conv1d_bn_relu_block_call<float, 32>(blk2_out, params["feat.fstn.conv3.weight"], params["feat.fstn.conv3.bias"], params["feat.fstn.bn3.weight"], params["feat.fstn.bn3.bias"], blk3_out, batch_size, truncated_length, 128, 1024); // feat.fstn.conv3.weight=[1024, 128], feat.fstn.conv3.bias=[1024], feat.fstn.bn3.running_mean=[1024], feat.fstn.bn3.running_var=[1024], feat.fstn.bn3.weight=[1024], feat.fstn.bn3.bias=[1024]

    maxReduce_call<float, 128>(blk3_out, tmp, blk4_out, batch_size, truncated_length, 1024);

    mlp_bn_relu_block_f32call<32>(blk4_out, params["feat.fstn.fc1.weight"], params["feat.fstn.fc1.bias"], params["feat.fstn.bn4.weight"], params["feat.fstn.bn4.bias"], blk5_out, batch_size, 1, 1024, 512); // feat.fstn.fc1.weight=[512, 1024], feat.fstn.fc1.bias=[512], feat.fstn.bn4.running_mean=[512], feat.fstn.bn4.running_var=[512], feat.fstn.bn4.weight=[512], feat.fstn.bn4.bias=[512]
    mlp_bn_relu_block_f32call<32>(blk5_out, params["feat.fstn.fc2.weight"], params["feat.fstn.fc2.bias"], params["feat.fstn.bn5.weight"], params["feat.fstn.bn5.bias"], blk6_out, batch_size, 1, 512, 256);  // feat.fstn.fc2.weight=[256, 512], feat.fstn.fc2.bias=[256], feat.fstn.bn5.running_mean=[256], feat.fstn.bn5.running_var=[256], feat.fstn.bn5.weight=[256], feat.fstn.bn5.bias=[256]

    LinearAddCoordBasis_call<float, 32>(blk6_out, params["feat.fstn.fc3.weight"], params["feat.fstn.fc3.bias"], blk10_out, batch_size, 1, 256, 4096, 64); // feat.fstn.fc3.weight=[1024, 256], feat.fstn.fc3.bias=[1024]

    // feat
    BatchGemm_call<float, 32>(blk9_out, blk10_out, blk11_out, batch_size, truncated_length, 64, 64);

    conv1d_bn_relu_block_call<float, 32>(blk11_out, params["feat.conv2.weight"], params["feat.conv2.bias"], params["feat.bn2.weight"], params["feat.bn2.bias"], blk2_out, batch_size, truncated_length, 64, 128);  // feat.conv2.weight=[128, 64], feat.conv2.bias=[128], feat.bn2.running_mean=[128], feat.bn2.running_var=[128], feat.bn2.weight=[128], feat.bn2.bias=[128]
    conv1d_bn_relu_block_call<float, 32>(blk2_out, params["feat.conv3.weight"], params["feat.conv3.bias"], params["feat.bn3.weight"], params["feat.bn3.bias"], blk3_out, batch_size, truncated_length, 128, 1024); // feat.conv3.weight=[1024, 128], feat.conv3.bias=[1024], feat.bn3.running_mean=[1024], feat.bn3.running_var=[1024], feat.bn3.weight=[1024], feat.bn3.bias=[1024]

    maxReduce_call<float, 128>(blk3_out, tmp, blk4_out, batch_size, truncated_length, 1024);

    // model
    mlp_bn_relu_block_f32call<32>(blk4_out, params["fc1.weight"], params["fc1.bias"], params["bn1.weight"], params["bn1.bias"], blk5_out, batch_size, 1, 1024, 512); // fc1.weight=[512, 1024], fc1.bias=[512], bn1.running_mean=[512], bn1.running_var=[512], bn1.weight=[512], bn1.bias=[512]

    mlp_bn_relu_block_f32call<32>(blk5_out, params["fc2.weight"], params["fc2.bias"], params["bn2.weight"], params["bn2.bias"], blk6_out, batch_size, 1, 512, 256); // fc2.weight=[256, 512], fc2.bias=[256], bn2.running_mean=[256], bn2.running_var=[256], bn2.weight=[256], bn2.bias=[256]

    Linear_call<float, 8>(blk6_out, params["fc3.weight"], params["fc3.bias"], blk12_out, batch_size, 1, 256, 10); // fc3.weight=[10, 256], fc3.bias=[10]

    LogSoftmax_10_call<float, 8>(blk12_out, batch_out, batch_size, 10);
}

void cudaFreeAll(float *blk_out[], int blk_num)
{
    for (int i = 1; i < blk_num; i++)
        CUDA_CHECK(cudaFree(blk_out[i]));
    for (int i = 0; i < 15; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

int main(int argc, char *argv[])
{
    // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
#ifdef DEBUG
    std::string dir = "/home/gpu_course/model_mean/txt"; // argv[1];
#endif
#ifndef DEBUG
    std::string dir = argv[1];
#endif
    set_weight_param_shape();
    // std::cout << argc << " " << dir;
    // 读取模型参数
    auto params = read_params(dir);
    preProcessBNLayer<32>(params);
    // 读取测试集数据
#ifdef DEBUG
    std::string file_path = "/home/gpu_course/data/test_point_clouds.h5";
#endif
#ifndef DEBUG
    std::string file_path = "./data/test_point_clouds.h5";
#endif
    int truncated_length = 16384, dataset_len = 1000;
    float *list_of_points = (float *)malloc(dataset_len * truncated_length * 3 * sizeof(float));
    int *list_of_labels = (int *)malloc(dataset_len * sizeof(int));
    if (list_of_labels == NULL || list_of_points == NULL)
    {
        std::cerr << "malloc failed" << std::endl;
        return -1;
    }
    dataset_len = read_h5_file(file_path, list_of_points, list_of_labels, truncated_length);
    // freopen("/home/gpu_course/out/v2_out.txt", "w", stdout);
    int batch_size = 96; // 96
    float *batch_data;
    int *dataset_lables;
    CUDA_CHECK(cudaMalloc(&batch_data, dataset_len * truncated_length * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dataset_lables, dataset_len * sizeof(int))); // 1024
    // TODO some blk_out may be reused
    float *blk1_out, *blk2_out, *blk3_out, *blk4_out, *blk5_out, *blk6_out, *blk7_out, *blk8_out, *blk9_out, *blk10_out, *blk11_out, *blk14_out, *tmp, *infer_out;
    int *infer_labels, *corrects;
    // stn
    CUDA_CHECK(cudaMalloc(&blk1_out, batch_size * truncated_length * 64 * sizeof(float)));   // relu(bn1d(conv1d(3,64)))
    CUDA_CHECK(cudaMalloc(&blk2_out, batch_size * truncated_length * 128 * sizeof(float)));  // relu(bn1d(conv1d(64,128)))
    CUDA_CHECK(cudaMalloc(&blk3_out, batch_size * truncated_length * 1024 * sizeof(float))); // relu(bn1d(conv1d(128,1024)))
    CUDA_CHECK(cudaMalloc(&blk4_out, batch_size * 1024 * sizeof(float)));                    // max(dim N)
    int blks = CEIL(MAX_LB_POWER2(truncated_length), 128);                                   // max reuduce blockNum
    CUDA_CHECK(cudaMalloc(&tmp, batch_size * blks * 1024 * sizeof(float)));                  // max_reduce_tmp, blockSize=128
    CUDA_CHECK(cudaMalloc(&blk5_out, batch_size * 512 * sizeof(float)));                     // mlp(1024,512)
    CUDA_CHECK(cudaMalloc(&blk6_out, batch_size * 256 * sizeof(float)));                     // mlp(512,256)
    CUDA_CHECK(cudaMalloc(&blk7_out, batch_size * 9 * sizeof(float)));                       // mlp(256,9)+add(9)
    // feat
    CUDA_CHECK(cudaMalloc(&blk8_out, batch_size * truncated_length * 3 * sizeof(float)));   // bmm(3,3)
    CUDA_CHECK(cudaMalloc(&blk9_out, batch_size * truncated_length * 64 * sizeof(float)));  // relu(bn1d(conv1d(3,64)))
    CUDA_CHECK(cudaMalloc(&blk11_out, batch_size * truncated_length * 64 * sizeof(float))); // bmm(64,64)
    // CUDA_CHECK(cudaMalloc(&blk12_out, batch_size * truncated_length * 128 * sizeof(float)));  // relu(bn1d(conv1d(64,128))) -> reuse blk2_out
    // CUDA_CHECK(cudaMalloc(&blk13_out, batch_size * truncated_length * 1024 * sizeof(float))); // relu(bn1d(conv1d(128,1024))) -> reuse blk3_out
    // fstn reuse stn
    CUDA_CHECK(cudaMalloc(&blk10_out, batch_size * 64 * 64 * sizeof(float))); // mlp(256,4096)+add(4096)
    // model
    CUDA_CHECK(cudaMalloc(&blk14_out, batch_size * 10 * sizeof(float)));  // mlp(256,10)
    CUDA_CHECK(cudaMalloc(&infer_out, dataset_len * 10 * sizeof(float))); // logsoftmax(10)
    CUDA_CHECK(cudaMalloc(&infer_labels, MAX_LB_POWER2(dataset_len) * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(infer_labels + dataset_len, 0, (MAX_LB_POWER2(dataset_len) * 2 - dataset_len) * sizeof(int)));
    float *blk_outs[] = {NULL, blk1_out, blk2_out, blk3_out, blk4_out, blk5_out, blk6_out, blk7_out, blk8_out, blk9_out, blk10_out, blk11_out, blk14_out};
    // 188M
    CUDA_CHECK(cudaMemcpy(batch_data, list_of_points, dataset_len * truncated_length * 3 * sizeof(float), cudaMemcpyHostToDevice));
    // 4K
    CUDA_CHECK(cudaMemcpy(dataset_lables, list_of_labels, dataset_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocManaged(&corrects, sizeof(int)));
    corrects[0] = 0;
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    size_t i;
    for (i = 0; i + batch_size <= dataset_len; i += batch_size)
        batch_infer<float>(params, batch_data + i * truncated_length * 3, blk_outs, tmp, infer_out + i * 10, truncated_length, batch_size);
    if (i < dataset_len)
        batch_infer<float>(params, batch_data + i * truncated_length * 3, blk_outs, tmp, infer_out + i * 10, truncated_length, dataset_len - i);
    argmaxWithLabels_call(infer_out, infer_labels, dataset_lables, dataset_len, 10);
    sumReduce_call<int>(infer_labels, corrects, dataset_len);
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // blk_outs[0] = tmp;
    // cudaFree(batch_data);
    // cudaFree(dataset_lables);
    // cudaFreeAll(blk_outs, 13);
    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << (float)(corrects[0]) / (float)dataset_len;
    return 0;
}