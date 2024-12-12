// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// #define DEBUG 1
#define MAX_LB_POWER2(x) ((x) > 0 ? (1 << (31 - __builtin_clz(x))) : 0)
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s %d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define CEIL(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define MAX_F16(a, b) __float2half(__builtin_fmaxf(__half2float((a)), __half2float((b))))
#define MUL_F16toF32(a, b) __half2float((a)) * __half2float((b))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])  // a vector of 4 floats
#define HALF4(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])   // a vector of 16 halfs
#define HALF8(pointer) FLOAT4(pointer)                               // a vector of 8 halfs
#define HALF16(pointer) (reinterpret_cast<double4 *>(&(pointer))[0]) // a vector of 16 halfs
#define ALIGNED64(x) (((x) + 63) & ~63)
#define ALIGNED32(x) (((x) + 31) & ~31)
#define ALIGNED16(x) (((x) + 15) & ~15)

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 700
#endif

// #if __CUDA_ARCH__ >= 700

std::unordered_map<std::string, std::vector<int>> weight_param_shape;
std::unordered_map<int, std::pair<std::string, int>> bn_layers_size;
cudaStream_t streams[15];

/****************************************************************************************
 * 参数预处理
 ****************************************************************************************/
void check_helper(half *dev, int B, int M, int Co)
{
    // if (dbg_print == 0)
    //     return;
    half *ck_blk = (half *)malloc((size_t)B * M * Co * sizeof(half));
    CUDA_CHECK(cudaMemcpy(ck_blk, dev, (size_t)B * M * Co * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B * M; i++)
    {
        for (int j = 0; j < Co; j++)
            printf("%.4f ", __half2float(ck_blk[i * Co + j]));
        printf("\n");
    }
    exit(0);
}
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
void set_bn_layers_size()
{
    bn_layers_size[0] = std::make_pair("bn2", 256);
    bn_layers_size[1] = std::make_pair("feat.stn.bn1", 64);
    bn_layers_size[2] = std::make_pair("feat.stn.bn2", 128);
    bn_layers_size[3] = std::make_pair("feat.stn.bn3", 1024);
    bn_layers_size[4] = std::make_pair("feat.stn.bn4", 512);
    bn_layers_size[5] = std::make_pair("feat.stn.bn5", 256);
    bn_layers_size[6] = std::make_pair("feat.bn1", 64);
    bn_layers_size[7] = std::make_pair("feat.fstn.bn1", 64);
    bn_layers_size[8] = std::make_pair("feat.fstn.bn2", 128);
    bn_layers_size[9] = std::make_pair("feat.fstn.bn3", 1024);
    bn_layers_size[10] = std::make_pair("feat.fstn.bn4", 512);
    bn_layers_size[11] = std::make_pair("feat.fstn.bn5", 256);
    bn_layers_size[12] = std::make_pair("feat.bn2", 128);
    bn_layers_size[13] = std::make_pair("feat.bn3", 1024);
    bn_layers_size[14] = std::make_pair("bn1", 512);
}
/***************************************************************************************
 * CUDA kernel functions
 ****************************************************************************************/
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
template <int blockSize = 32>
__always_inline void transpose2D_f16_call(half *input, half *output, int M, int N)
{
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL(N, blockSize), CEIL(M, blockSize));
    transpose2D<half, blockSize><<<gridDim, blockDim>>>(input, output, M, N);
}
/*********************************************************************
 * Gemm
 *********************************************************************/
/**
 * @brief preprocess Conv1d and BN param
 */
__global__ void PreprocessBN_f16(half *weight, half *bias, half *running_mean, half *running_var, int in_channel)
{
    // weight <- weight / sqrt(var+eps)
    // bias   <- bias - mean * weight / sqrt(var+eps)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= in_channel)
        return;
    float var_f32 = __half2float(running_var[x]), mean_f32 = __half2float(running_mean[x]);
    float w_f32 = __half2float(weight[x]), b_f32 = __half2float(bias[x]);
    float alpha = w_f32 * __frsqrt_rn(var_f32 + 1e-5);
    weight[x] = alpha;
    bias[x] = b_f32 - mean_f32 * alpha;
}
__always_inline void PreprocessBN_f16_call(half *weight, half *bias, half *running_mean, half *running_var, int channel, cudaStream_t stream)
{
    dim3 blockDim(64);
    dim3 gridDim(CEIL(channel, 64));
    PreprocessBN_f16<<<gridDim, blockDim, 0, stream>>>(weight, bias, running_mean, running_var, channel);
}
/**
 * @brief Tiled matrix multiplication kernel AB+v
 * @attention v.shape=[K]
 * @param Md: M * N
 * @param Nd: N * K
 * @param Pd: M * K
 */
template <int TILED_SIZE = 16>
__global__ void TiledGemmWithBias_BN_ReLU_fp16(half *a, half *b, half *c, half *v, half *alpha, half *beta, int M, int N, int K)
{
    constexpr int PAD = 8;
    __shared__ half s_a[TILED_SIZE][TILED_SIZE];
    __shared__ half s_b[TILED_SIZE][TILED_SIZE + PAD];
    __shared__ half s_v[TILED_SIZE];
    __shared__ half s_alpha[TILED_SIZE];
    __shared__ half s_beta[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0 && x < N)
    {
        s_v[threadIdx.x] = v[x];
        s_alpha[threadIdx.x] = alpha[x];
        s_beta[threadIdx.x] = beta[x];
    }
    __syncthreads();
    float Pvalue = __half2float(s_v[threadIdx.x]);
#pragma unroll 4
    for (int i = 0; i < K; i += TILED_SIZE)
    {
        if (i + threadIdx.x >= K || y >= M)
            s_a[threadIdx.y][threadIdx.x] = 0.0;
        else
            s_a[threadIdx.y][threadIdx.x] = a[y * K + i + threadIdx.x];
        if (i + threadIdx.y >= K || x >= N)
            s_b[threadIdx.y][threadIdx.x] = 0.0;
        else
            s_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + x];

        __syncthreads(); // 等待所有线程加载完毕
#pragma unroll
        for (int k = 0; k < TILED_SIZE; ++k)
            Pvalue += MUL_F16toF32(s_a[threadIdx.y][k], s_b[k][threadIdx.x]);
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= N)
        return;
    Pvalue = Pvalue * __half2float(s_alpha[threadIdx.x]) + __half2float(s_beta[threadIdx.x]);
    Pvalue = Pvalue > 0.0 ? Pvalue : 0.0;
    c[y * N + x] = __half2float(Pvalue);
}
template <int TILED_SIZE = 16>
__always_inline void GemmWithBias_BN_ReLU_fp16_call(half *in, half *weight, half *bias, half *alpha, half *beta, half *out, int M, int Ci, int Co)
{
    dim3 blockDim(TILED_SIZE, TILED_SIZE);
    dim3 gridDim(CEIL(Co, TILED_SIZE), CEIL(M, TILED_SIZE));
    TiledGemmWithBias_BN_ReLU_fp16<TILED_SIZE><<<gridDim, blockDim>>>(in, weight, out, bias, alpha, beta, M, Co, Ci);
}
template <int TILED_SIZE = 16>
__global__ void TiledGemmWithBias_fp16(half *a, half *b, half *c, half *v, int M, int N, int K)
{
    constexpr int PAD = 8;
    __shared__ half Mds[TILED_SIZE][TILED_SIZE];
    __shared__ half Nds[TILED_SIZE][TILED_SIZE + PAD];
    __shared__ half vs[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0)
        vs[threadIdx.x] = x >= N ? __float2half(0.0) : v[x];
    __syncthreads();
    float Pvalue = __float2half(vs[threadIdx.x]);
#pragma unroll 4
    for (int i = 0; i < K; i += TILED_SIZE)
    {
        if (i + threadIdx.x >= K || y >= M)
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Mds[threadIdx.y][threadIdx.x] = a[y * K + i + threadIdx.x];
        if (i + threadIdx.y >= K || x >= N)
            Nds[threadIdx.y][threadIdx.x] = 0.0;
        else
            Nds[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + x];

        __syncthreads(); // 等待所有线程加载完毕
#pragma unroll
        for (int k = 0; k < TILED_SIZE; ++k)
            Pvalue += MUL_F16toF32(Mds[threadIdx.y][k], Nds[k][threadIdx.x]);
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= N)
        return;
    c[y * N + x] = __half2float(Pvalue);
}
template <int TILED_SIZE = 16>
__always_inline void GemmWithBias_fp16_call(half *in, half *weight, half *bias, half *out, int M, int Ci, int Co)
{
    dim3 blockDim(TILED_SIZE, TILED_SIZE);
    dim3 gridDim(CEIL(Co, TILED_SIZE), CEIL(M, TILED_SIZE));
    TiledGemmWithBias_fp16<TILED_SIZE><<<gridDim, blockDim>>>(in, weight, out, bias, M, Co, Ci);
}

/**
 * @brief Based on tiled(register-level) Gemm
 */
__global__ void m512n64k3_bn_relu_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, half *__restrict__ alpha, half *__restrict__ beta, const int M, const int N, const int K)
{
    // BK==K
    constexpr const int BK = 3, TM = 16, TN = 8;
    /*
        需要从A中加载BM*BK个元素, 使用HALF16+HALF8加载，共需要BM*BK/24=64个线程
        从B中加载BK*BN个元素, 使用HALF8加载，共需要BN*BK/8=24个线程

        需要计算BM*BN个C中的元素，每个线程负责TM*TN个元素
        共需要BM*BN/(TM*TN)=256个线程
    */
    const int BM = 512, BN = 64;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    constexpr int load_a_unit1 = 16;
    constexpr int load_a_unit2 = 8;
    constexpr int load_b_unit = 8;
    constexpr int thrdn_load_a = BM * BK / (load_a_unit1 + load_a_unit2);
    constexpr int thrdn_load_b = BN * BK / load_b_unit;
    constexpr int thrdn_load_v = BN / 8;

    constexpr int PAD = 8;
    __shared__ half s_a[BK * BM];      // aligned to 4B
    __shared__ half s_b[BK][BN + PAD]; // aligned to 4B
    __shared__ half s_v[BN];
    __shared__ half s_alpha[BN];
    __shared__ half s_beta[BN];

    // half r_load_a[load_a_unit1]; // load_a_unit1 = max(load_a_unit1, load_a_unit2)
    // half r_load_b[load_b_unit];

    half r_comp_a[TM];
    half r_comp_b[TN];
    half r_c[TM][TN] = {0.0};

    int load_a_smem_addr1 = tid * load_a_unit1;
    int load_a_smem_addr2 = tid * load_a_unit2 + thrdn_load_a * load_a_unit1;

    int load_b_smem_k = tid / (BN / load_b_unit);
    int load_b_smem_n = (tid % (BN / load_b_unit)) * load_b_unit;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (tid < thrdn_load_a)
    {
        int load_a_gmem_addr1 = by * BM * K + load_a_smem_addr1;
        HALF16(s_a[load_a_smem_addr1]) = HALF16(a[load_a_gmem_addr1]);
        int load_a_gmem_addr2 = by * BM * K + load_a_smem_addr2;
        HALF8(s_a[load_a_smem_addr2]) = HALF8(a[load_a_gmem_addr2]);
        if (tid < thrdn_load_b)
        {
            int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, BN);
            HALF8(s_b[load_b_smem_k][load_b_smem_n]) = HALF8(b[load_b_gmem_addr]);
        }
        if (tid < thrdn_load_v)
        {
            HALF8(s_v[load_b_smem_n]) = HALF8(v[load_b_gmem_n]);
            HALF8(s_alpha[load_b_smem_n]) = HALF8(alpha[load_b_gmem_n]);
            HALF8(s_beta[load_b_smem_n]) = HALF8(beta[load_b_gmem_n]);
        }
    }
    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++)
    {
#pragma unroll
        for (int i = 0; i < TM; i++)
            r_comp_a[i] = s_a[OFFSET(ty * TM + i, tk, BK)];
        HALF8(r_comp_b[0]) = HALF8(s_b[tk][tx * TN]);
#pragma unroll
        for (int tm = 0; tm < TM; tm++)
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                r_c[tm][tn] = __float2half(__half2float(r_c[tm][tn]) + MUL_F16toF32(r_comp_a[tm], r_comp_b[tn]));
            }
    }

#pragma unroll
    for (int tm = 0; tm < TM; tm++)
#pragma unroll
        for (int tn = 0; tn < TN; tn++)
        {
            int col = tx * TN + tn;
            float tmp = __half2float(r_c[tm][tn]) + __half2float(s_v[col]);     // fusion bias
            tmp = tmp * __half2float(s_alpha[col]) + __half2float(s_beta[col]); // fusion preprocessed relu
            tmp = tmp > 0.0 ? tmp : 0.0;                                        // fusion relu
            r_c[tm][tn] = __float2half(tmp);
        }
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        HALF8(c[store_c_gmem_addr]) = HALF8(r_c[i][0]);
    }
}
__always_inline void m512n64k3_bn_relu_fp16_call(half *a, half *b, half *v, half *alpha, half *beta, half *c, int B, int M, int Ci, int Co)
{
    int m = B * M;
    int n = Co;
    int k = Ci;
    constexpr int BM = 512, BN = 64, TM = 16, TN = 8;
    dim3 blockDim(CEIL(BN, TN), CEIL(BM, TM));
    dim3 gridDim(CEIL(n, BN), CEIL(m, BM));
    m512n64k3_bn_relu_fp16<<<gridDim, blockDim>>>(a, b, c, v, alpha, beta, m, n, k);
}
/**
 * @brief Based on wmma, Tiled matrix multiplication kernel AB+v
 * @attention M % 256 == 0, N % 128 == 0, K % 32 == 0
 * @param a: M * K
 * @param b: K * N
 * @param c: M * N
 *
 */
__global__ void m256n128k32_bn_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, half *alpha, half *beta, const int M, const int N, const int K)
{

    const int BM = 256;
    const int BN = 128;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int laneid = tid & 31;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];
    __shared__ half s_v[BN];
    __shared__ half s_alpha[BN];
    __shared__ half s_beta[BN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 2;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 4) << 1;
    int load_b_smem_n = (tid & 15) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid >> 1;
    int comp_c_frag_n = wid & 1;

    if (tid < (BN / 8))
    {
        int load_v_gmem_addr = bx * BN + load_b_smem_n;
        HALF8(s_v[load_b_smem_n]) = HALF8(v[load_v_gmem_addr]);
        HALF8(s_alpha[load_b_smem_n]) = HALF8(alpha[load_v_gmem_addr]);
        HALF8(s_beta[load_b_smem_n]) = HALF8(beta[load_v_gmem_addr]);
    }

    for (int bk = 0; bk < K / BK; bk++)
    {
        HALF8(s_a[load_a_smem_m][load_a_smem_k]) = HALF8(a[load_a_gmem_addr]);
        HALF8(s_a[load_a_smem_m + 1][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + K]);
        HALF8(s_a[load_a_smem_m + 2][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 2 * K]);
        HALF8(s_a[load_a_smem_m + 3][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 3 * K]);
        HALF8(s_b[load_b_smem_k][load_b_smem_n]) = HALF8(b[load_b_gmem_addr]);
        HALF8(s_b[load_b_smem_k + 1][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
#pragma unroll
            for (int t = 0; t < frag_c[i][j].num_elements; t++)
            {
                int col = comp_c_frag_n * 64 + j * 16 + t + ((laneid / 8) & 1) * 8;
                float tmp = __half2float(frag_c[i][j].x[t]) + __half2float(s_v[col]); // fusion bias
                tmp = tmp * __half2float(s_alpha[col]) + __half2float(s_beta[col]);   // fusion bn
                frag_c[i][j].x[t] = __float2half(tmp);
            }
        }
    }

    size_t store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    size_t store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
__always_inline void m256n128k32_bn_fp16_call(half *a, half *b, half *v, half *alpha, half *beta, half *c, int B, int M, int Ci, int Co)
{
    const int BM = 256, BN = 128;
    dim3 blockDim(256);
    M = B * M;
    dim3 gridDim(CEIL(Co, BN), CEIL(M, BM));
    m256n128k32_bn_fp16<<<gridDim, blockDim>>>(a, b, c, v, alpha, beta, M, Co, Ci);
}
__global__ void m256n128k32_bn_relu_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, half *alpha, half *beta, const int M, const int N, const int K)
{
    const int BM = 256;
    const int BN = 128;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int laneid = tid & 31;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];
    __shared__ half s_v[BN];
    __shared__ half s_alpha[BN];
    __shared__ half s_beta[BN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 2;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 4) << 1;
    int load_b_smem_n = (tid & 15) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid >> 1;
    int comp_c_frag_n = wid & 1;

    if (tid < (BN / 8))
    {
        int load_v_gmem_addr = bx * BN + load_b_smem_n;
        HALF8(s_v[load_b_smem_n]) = HALF8(v[load_v_gmem_addr]);
        HALF8(s_alpha[load_b_smem_n]) = HALF8(alpha[load_v_gmem_addr]);
        HALF8(s_beta[load_b_smem_n]) = HALF8(beta[load_v_gmem_addr]);
    }

    for (int bk = 0; bk < K / BK; bk++)
    {
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr + K]);
        FLOAT4(s_a[load_a_smem_m + 2][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr + 2 * K]);
        FLOAT4(s_a[load_a_smem_m + 3][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr + 3 * K]);

        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
#pragma unroll
            for (int t = 0; t < frag_c[i][j].num_elements; t++)
            {
                int col = comp_c_frag_n * 64 + j * 16 + t + ((laneid / 8) & 1) * 8;
                float tmp = __half2float(frag_c[i][j].x[t]) + __half2float(s_v[col]); // fusion bias
                tmp = tmp * __half2float(s_alpha[col]) + __half2float(s_beta[col]);   // fusion bn
                tmp = tmp > 0.0 ? tmp : 0.0;                                          // fusion relu
                frag_c[i][j].x[t] = __float2half(tmp);
            }
        }
    }

    size_t store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    size_t store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
__always_inline void m256n128k32_bn_relu_fp16_call(half *a, half *b, half *v, half *alpha, half *beta, half *c, int B, int M, int Ci, int Co)
{
    const int BM = 256, BN = 128;
    dim3 blockDim(256);
    M = B * M;
    dim3 gridDim(CEIL(Co, BN), CEIL(M, BM));
    m256n128k32_bn_relu_fp16<<<gridDim, blockDim>>>(a, b, c, v, alpha, beta, M, Co, Ci);
}

__global__ void m256n64k32_bn_relu_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, half *__restrict__ alpha, half *__restrict__ beta, const int M, const int N, const int K)
{
    const int BM = 256;
    const int BN = 64;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid & 31;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];
    __shared__ half s_v[BN];
    __shared__ half s_alpha[BN];
    __shared__ half s_beta[BN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    /*
        每32个线程计算c中一个(16*4)*(16*4)的块
        256*64的c需要计算256*64/64/64=4个块, 也就是4*32=128个线程
        每个线程使用HALF8加载
        对于a, 需要加载BM*BK个元素, 每个线程需要加载BM*BK/128/8=8个HALF8, 每个线程在连续的4行中加载
        对于b, 需要加载BK*BN个元素, 每个线程需要加载BK*BN/128/8=2个HALF8
     */
    int load_a_smem_m = (tid >> 2) << 3; // tid/4*8
    int load_a_smem_k = (tid & 3) << 3;  // tid%4*8
    int load_b_smem_k = (tid >> 3) << 1; // tid/8*2
    int load_b_smem_n = (tid & 7) << 3;  // tid%8*8

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid;
    int comp_c_frag_n = 0;

    if (tid < (BN / 8))
    {
        int load_v_gmem_addr = bx * BN + load_b_smem_n;
        HALF8(s_v[load_b_smem_n]) = HALF8(v[load_v_gmem_addr]);
        HALF8(s_alpha[load_b_smem_n]) = HALF8(alpha[load_v_gmem_addr]);
        HALF8(s_beta[load_b_smem_n]) = HALF8(beta[load_v_gmem_addr]);
    }

    for (int bk = 0; bk < K / BK; bk++)
    {
        HALF8(s_a[load_a_smem_m][load_a_smem_k]) = HALF8(a[load_a_gmem_addr]);
        HALF8(s_a[load_a_smem_m + 1][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + K]);
        HALF8(s_a[load_a_smem_m + 2][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 2 * K]);
        HALF8(s_a[load_a_smem_m + 3][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 3 * K]);
        HALF8(s_a[load_a_smem_m + 4][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 4 * K]);
        HALF8(s_a[load_a_smem_m + 5][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 5 * K]);
        HALF8(s_a[load_a_smem_m + 6][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 6 * K]);
        HALF8(s_a[load_a_smem_m + 7][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 7 * K]);
        HALF8(s_b[load_b_smem_k][load_b_smem_n]) = HALF8(b[load_b_gmem_addr]);
        HALF8(s_b[load_b_smem_k + 1][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            for (int t = 0; t < frag_c[i][j].num_elements; t++)
            {
                int col = comp_c_frag_n * 64 + j * 16 + t + ((laneid / 8) & 1) * 8;
                float tmp = __half2float(frag_c[i][j].x[t]) + __half2float(s_v[col]); // fusion bias
                tmp = tmp * __half2float(s_alpha[col]) + __half2float(s_beta[col]);   // fusion bn
                tmp = tmp > 0.0f ? tmp : 0.0f;                                        // fusion relu
                frag_c[i][j].x[t] = __float2half(tmp);
            }
        }
    }

    size_t store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    size_t store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
__always_inline void m256n64k32_bn_relu_fp16_call(half *a, half *b, half *v, half *alpha, half *beta, half *c, int B, int M, int Ci, int Co)
{
    const int BM = 256, BN = 64;
    dim3 blockDim(128);
    M = B * M;
    int BX = (Co + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY);
    m256n64k32_bn_relu_fp16<<<gridDim, blockDim>>>(a, b, c, v, alpha, beta, M, Co, Ci);
}
__global__ void m64n256k32_bn_relu_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, half *__restrict__ alpha, half *__restrict__ beta, const int M, const int N, const int K)
{
    const int BM = 64;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid & 31;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];
    __shared__ half s_v[BN];
    __shared__ half s_alpha[BN];
    __shared__ half s_beta[BN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    /*
        每32个线程计算c中一个(16*4)*(16*4)的块
        64*256的c需要计算64*256/64/64=4个块, 也就是4*32=128个线程
        每个线程使用HALF8加载
        对于a, 需要加载BM*BK个元素, 每个线程需要加载BM*BK/128/8=2个HALF8, 每个线程在连续的2行中加载
        对于b, 需要加载BK*BN个元素, 每个线程需要加载BK*BN/128/8=8个HALF8
     */
    int load_a_smem_m = (tid >> 2) << 1; // tid/4*2
    int load_a_smem_k = (tid & 3) << 3;  // tid%4*8
    int load_b_smem_k = (tid >> 5) << 3; // tid/32*8
    int load_b_smem_n = (tid & 31) << 3; // tid%32*8

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = 0;
    int comp_c_frag_n = wid;

    if (tid < (BN / 8))
    {
        int load_v_gmem_addr = bx * BN + load_b_smem_n;
        HALF8(s_v[load_b_smem_n]) = HALF8(v[load_v_gmem_addr]);
        HALF8(s_alpha[load_b_smem_n]) = HALF8(alpha[load_v_gmem_addr]);
        HALF8(s_beta[load_b_smem_n]) = HALF8(beta[load_v_gmem_addr]);
    }

    for (int bk = 0; bk < K / BK; bk++)
    {
        HALF8(s_a[load_a_smem_m][load_a_smem_k]) = HALF8(a[load_a_gmem_addr]);
        HALF8(s_a[load_a_smem_m + 1][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + K]);

        HALF8(s_b[load_b_smem_k][load_b_smem_n]) = HALF8(b[load_b_gmem_addr]);
        HALF8(s_b[load_b_smem_k + 1][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + N]);
        HALF8(s_b[load_b_smem_k + 2][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 2 * N]);
        HALF8(s_b[load_b_smem_k + 3][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 3 * N]);
        HALF8(s_b[load_b_smem_k + 4][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 4 * N]);
        HALF8(s_b[load_b_smem_k + 5][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 5 * N]);
        HALF8(s_b[load_b_smem_k + 6][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 6 * N]);
        HALF8(s_b[load_b_smem_k + 7][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 7 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            for (int t = 0; t < frag_c[i][j].num_elements; t++)
            {
                int col = comp_c_frag_n * 64 + j * 16 + t + ((laneid / 8) & 1) * 8;
                /*
                    dispaly the layout of frag_c
                    +---------------+...+---------------+
                    | 0-3   | 8-11  |   | 0-3   | 8-11  |
                    | 16-19 | 24-27 |   | 16-19 | 24-27 |
                    | 4-7   | 12-15 |   | 4-7   | 12-15 |
                    | 20-23 | 28-31 |   | 20-23 | 28-31 |
                    +---------------+...+---------------+
                    |      ...      |...|      ...      |
                    +---------------+...+---------------+
                    | 0-3   | 8-11  |   | 0-3   | 8-11  |
                    | 16-19 | 24-27 |   | 16-19 | 24-27 |
                    | 4-7   | 12-15 |   | 4-7   | 12-15 |
                    | 20-23 | 28-31 |   | 20-23 | 28-31 |
                    +---------------+...+---------------+
                 */
                float tmp = __half2float(frag_c[i][j].x[t]) + __half2float(s_v[col]); // fusion bias
                tmp = tmp * __half2float(s_alpha[col]) + __half2float(s_beta[col]);   // fusion bn
                tmp = tmp > 0.0 ? tmp : 0.0;                                          // fusion relu
                frag_c[i][j].x[t] = __float2half(tmp);
            }
        }
    }

    size_t store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    size_t store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
__always_inline void m64n256k32_bn_relu_fp16_call(half *a, half *b, half *v, half *alpha, half *beta, half *c, int B, int M, int Ci, int Co)
{
    const int BM = 64, BN = 256;
    dim3 blockDim(128);
    M = B * M;
    int BX = CEIL(Co, BN);
    int BY = CEIL(M, BM);
    dim3 gridDim(BX, BY);

    m64n256k32_bn_relu_fp16<<<gridDim, blockDim>>>(a, b, c, v, alpha, beta, M, Co, Ci);
}
__always_inline void Linear_BN_ReLU_fp16_call(half *in, half *weight, half *bias, half *alpha, half *beta, half *out, int B, int M, int Ci, int Co)
{
    int m = B * M;
    if (m % 64 != 0 || Co % 64 != 0 || Ci % 32 != 0)
    {
        if (Ci == 3 && m >= 512 && m % 512 == 0)
            m512n64k3_bn_relu_fp16_call(in, weight, bias, alpha, beta, out, B, M, Ci, Co);
        else
            GemmWithBias_BN_ReLU_fp16_call(in, weight, bias, alpha, beta, out, B * M, Ci, Co);
        return;
    }
    if (m >= 256 && Co >= 128)
        m256n128k32_bn_relu_fp16_call(in, weight, bias, alpha, beta, out, B, M, Ci, Co);
    else if (m >= 256)
        m256n64k32_bn_relu_fp16_call(in, weight, bias, alpha, beta, out, B, M, Ci, Co);
    else if (Co >= 256)
        m64n256k32_bn_relu_fp16_call(in, weight, bias, alpha, beta, out, B, M, Ci, Co);
    else
        GemmWithBias_BN_ReLU_fp16_call(in, weight, bias, alpha, beta, out, B * M, Ci, Co);
}
__always_inline void Conv1d_BN_ReLU_fp16_call(half *in, half *weight, half *bias, half *alpha, half *beta, half *out, int B, int M, int Ci, int Co)
{
    Linear_BN_ReLU_fp16_call(in, weight, bias, alpha, beta, out, B, M, Ci, Co);
}
template <int TILED_SIZE = 16>
__global__ void m64n9k256AddBasis_fp16(half *a, half *b, half *c, half *v, int M, int N, int K)
{
    constexpr int PAD = 8;
    __shared__ half s_a[TILED_SIZE][TILED_SIZE];
    __shared__ half s_b[TILED_SIZE][TILED_SIZE + PAD];
    __shared__ half s_v[TILED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.y == 0 && x < N)
    {
        s_v[threadIdx.x] = v[x] + __float2half((x % 3 == x / 3) ? 1.0 : 0.0);
    }
    __syncthreads();
    float Pvalue = __half2float(s_v[threadIdx.x]);
#pragma unroll 4
    for (int i = 0; i < K; i += TILED_SIZE)
    {
        if (i + threadIdx.x >= K || y >= M)
            s_a[threadIdx.y][threadIdx.x] = 0.0;
        else
            s_a[threadIdx.y][threadIdx.x] = a[y * K + i + threadIdx.x];
        if (i + threadIdx.y >= K || x >= N)
            s_b[threadIdx.y][threadIdx.x] = 0.0;
        else
            s_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + x];

        __syncthreads(); // 等待所有线程加载完毕
#pragma unroll
        for (int k = 0; k < TILED_SIZE; ++k)
            Pvalue += MUL_F16toF32(s_a[threadIdx.y][k], s_b[k][threadIdx.x]);
        __syncthreads(); // 防止下一次迭代时，数据被覆盖
    }
    if (y >= M || x >= N)
        return;
    c[y * N + x] = __half2float(Pvalue);
}
template <int TILED_SIZE = 16>
__always_inline void m64n9k256AddBasis_fp16_call(half *in, half *weight, half *bias, half *out, int M, int Ci, int Co)
{
    dim3 blockDim(TILED_SIZE, TILED_SIZE);
    dim3 gridDim(CEIL(Co, TILED_SIZE), CEIL(M, TILED_SIZE));
    m64n9k256AddBasis_fp16<TILED_SIZE><<<gridDim, blockDim>>>(in, weight, out, bias, M, Co, Ci);
}
__global__ void m64n4096k256AddBasis_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, const int M, const int N, const int K)
{
    const int BM = 64;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid & 31;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];
    __shared__ half s_v[BN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    /*
        每32个线程计算c中一个(16*4)*(16*4)的块
        64*256的c需要计算64*256/64/64=4个块, 也就是4*32=128个线程
        每个线程使用HALF8加载
        对于a, 需要加载BM*BK个元素, 每个线程需要加载BM*BK/128/8=2个HALF8, 每个线程在连续的2行中加载
        对于b, 需要加载BK*BN个元素, 每个线程需要加载BK*BN/128/8=8个HALF8
     */
    int load_a_smem_m = (tid >> 2) << 1; // tid/4*2
    int load_a_smem_k = (tid & 3) << 3;  // tid%4*8
    int load_b_smem_k = (tid >> 5) << 3; // tid/32*8
    int load_b_smem_n = (tid & 31) << 3; // tid%32*8

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = 0;
    int comp_c_frag_n = wid;

    if (tid < (BN / 8))
    {
        int load_v_gmem_addr = bx * BN + load_b_smem_n;
        HALF8(s_v[load_b_smem_n]) = HALF8(v[load_v_gmem_addr]);
    }

    for (int bk = 0; bk < K / BK; bk++)
    {
        HALF8(s_a[load_a_smem_m][load_a_smem_k]) = HALF8(a[load_a_gmem_addr]);
        HALF8(s_a[load_a_smem_m + 1][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + K]);

        HALF8(s_b[load_b_smem_k][load_b_smem_n]) = HALF8(b[load_b_gmem_addr]);
        HALF8(s_b[load_b_smem_k + 1][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + N]);
        HALF8(s_b[load_b_smem_k + 2][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 2 * N]);
        HALF8(s_b[load_b_smem_k + 3][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 3 * N]);
        HALF8(s_b[load_b_smem_k + 4][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 4 * N]);
        HALF8(s_b[load_b_smem_k + 5][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 5 * N]);
        HALF8(s_b[load_b_smem_k + 6][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 6 * N]);
        HALF8(s_b[load_b_smem_k + 7][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + 7 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
#pragma unroll
            for (int t = 0; t < frag_c[i][j].num_elements; t++)
            {
                int col = comp_c_frag_n * 64 + j * 16 + t + ((laneid / 8) & 1) * 8;
                if (((col + bx * BN) & 63) == ((col + bx * BN) >> 6)) // add basis
                    frag_c[i][j].x[t] = __half2float(frag_c[i][j].x[t]) + 1.0;
                frag_c[i][j].x[t] = __hadd(frag_c[i][j].x[t], s_v[col]);
            }
        }
    }

    size_t store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    size_t store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
__always_inline void m64n4096k256AddBasis_fp16_call(half *in, half *weight, half *bias, half *out, int M, int Ci, int Co)
{
    const int BM = 64, BN = 256;
    dim3 blockDim(128);
    int BX = CEIL(Co, BN);
    int BY = CEIL(M, BM);
    dim3 gridDim(BX, BY);
    m64n4096k256AddBasis_fp16<<<gridDim, blockDim>>>(in, weight, out, bias, M, Co, Ci);
}
/*********************************************************************
 * Batch Gemm
 *********************************************************************/
__global__ void bm512n3k3_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, const int B, const int M, const int N, const int K)
{
    // BK==K
    constexpr const int BM = 512, BN = 3, BK = 3, TM = 16, TN = 3;
    /*
        需要从A中加载BM*BK个元素, 使用HALF16*3加载，共需要BM*BK/48(=32)个线程
        从B中加载BK*BN个元素, 使用HALF1加载，共需要BN*BK/1(=9)个线程

        需要计算BM*BN个C中的元素，每个线程负责TM*TN个元素
        共需要BM/TM*BN/*TN(=32)个线程
    */
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    constexpr int load_a_unit = 16;
    constexpr int store_c_unit = 16;
    constexpr int load_b_unit = 1;
    constexpr int thrdn_load_b = BN * BK / load_b_unit;
    constexpr int aligned_rows = load_a_unit / BK;
    constexpr int round = TM / aligned_rows;

    __shared__ half s_b[ALIGNED16(BK * BN)]; // aligned to 4B

    half r_c[ALIGNED32(TM * TN)] = {0.0};          // 128B
    half r_comp_a[ALIGNED32(load_a_unit * round)]; // 128B
    half r_comp_b[ALIGNED16(BK * BN)];             // 32B

    int a_batch_offset = bz * M * K;
    int b_batch_offset = bz * N * K;
    int c_batch_offset = bz * M * N;
    int load_a_gmem_addr = (by * BM + tid * TM) * K + a_batch_offset;
    int store_c_a_gmem_m = by * BM + tid * TM;
    int load_b_smem_n = tid % 3;
    int load_b_smem_k = tid / 3;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_b_smem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, BN);
    // 加载b
    if (tid < thrdn_load_b)
    {
        int load_b_gmem_addr = b_batch_offset + load_b_smem_addr;
        s_b[load_b_smem_addr] = b[load_b_gmem_addr];
    }
    // 各个线程加载各自的第一个unit
    {
        HALF16(r_comp_a[0]) = HALF16(a[load_a_gmem_addr]);
    }
    // 直接把share memory中的数据加载到寄存器中
    HALF16(r_comp_b[0]) = HALF16(s_b[0]);
    __syncthreads();
    int bk;
    for (bk = 1; bk < round; bk++)
    {
        // 加载下一次的a
        HALF16(r_comp_a[load_a_unit * bk]) = HALF16(a[load_a_gmem_addr + load_a_unit * bk]);
        // 计算上一次的smem数据
        for (int i = 0; i < aligned_rows; i++)
        {
            int row = (bk - 1) * aligned_rows + i;
            for (int k = 0; k < BK; k++)
            {
                for (int j = 0; j < TN; j++)
                {
                    r_c[row * TN + j] = __hadd(r_c[row * TN + j], __hmul(r_comp_a[row * BK + k], r_comp_b[k * BK + j]));
                }
            }
        }
    }
    {
        for (int i = 0; i < TM - (bk - 1) * aligned_rows; i++)
        {
            int row = (bk - 1) * aligned_rows + i;
            for (int k = 0; k < BK; k++)
            {
                for (int j = 0; j < TN; j++)
                {
                    r_c[row * TN + j] = __hadd(r_c[row * TN + j], __hmul(r_comp_a[row * BK + k], r_comp_b[k * BK + j]));
                }
            }
        }
    }
    int store_c_gmem_addr = c_batch_offset + store_c_a_gmem_m * N;
#pragma unroll
    for (int i = 0; i < TM * TN; i += store_c_unit)
    {
        HALF16(c[store_c_gmem_addr]) = HALF16(r_c[i]);
        store_c_gmem_addr += store_c_unit;
    }
}
__always_inline void bm512n3k3_fp16_call(half *a, half *b, half *c, int B, int M, int N, int K)
{
    constexpr int BM = 512, BN = 3;
    dim3 blockDim(32);
    dim3 gridDim(CEIL(N, BN), CEIL(M, BM), B);
    bm512n3k3_fp16<<<gridDim, blockDim>>>(a, b, c, B, M, N, K);
}
__global__ void bm256n64k32_fp16(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, const int M, const int N, const int K)
{
    const int BM = 256;
    const int BN = 64;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tid = threadIdx.x;
    int wid = tid / 32;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    /*
        每32个线程计算c中一个(16*4)*(16*4)的块
        256*64的c需要计算256*64/64/64=4个块, 也就是4*32=128个线程
        每个线程使用HALF8加载
        对于a, 需要加载BM*BK个元素, 每个线程需要加载BM*BK/128/8=8个HALF8, 每个线程在连续的4行中加载
        对于b, 需要加载BK*BN个元素, 每个线程需要加载BK*BN/128/8=2个HALF8
     */
    int load_a_smem_m = (tid >> 2) << 3; // tid/4*8
    int load_a_smem_k = (tid & 3) << 3;  // tid%4*8
    int load_b_smem_k = (tid >> 3) << 1; // tid/8*2
    int load_b_smem_n = (tid & 7) << 3;  // tid%8*8

    int load_a_gmem_m = by * BM + bz * M + load_a_smem_m; // 相当于越过bz*M行
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k + bz * K, load_b_gmem_n, N); // 相当于越过bz*K行

    int comp_c_frag_m = wid;
    int comp_c_frag_n = 0;

    for (int bk = 0; bk < K / BK; bk++)
    {
        HALF8(s_a[load_a_smem_m][load_a_smem_k]) = HALF8(a[load_a_gmem_addr]);
        HALF8(s_a[load_a_smem_m + 1][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + K]);
        HALF8(s_a[load_a_smem_m + 2][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 2 * K]);
        HALF8(s_a[load_a_smem_m + 3][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 3 * K]);
        HALF8(s_a[load_a_smem_m + 4][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 4 * K]);
        HALF8(s_a[load_a_smem_m + 5][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 5 * K]);
        HALF8(s_a[load_a_smem_m + 6][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 6 * K]);
        HALF8(s_a[load_a_smem_m + 7][load_a_smem_k]) = HALF8(a[load_a_gmem_addr + 7 * K]);
        HALF8(s_b[load_b_smem_k][load_b_smem_n]) = HALF8(b[load_b_gmem_addr]);
        HALF8(s_b[load_b_smem_k + 1][load_b_smem_n]) = HALF8(b[load_b_gmem_addr + N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                nvcuda::wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
        __syncthreads();
    }

    size_t store_c_gmem_m = bz * M + by * BM + comp_c_frag_m * 64;
    size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    size_t store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            nvcuda::wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
}
__always_inline void bm256n64k32_fp16_call(half *a, half *b, half *c, int B, int M, int N, int K)
{
    const int BM = 256, BN = 64;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY, B);
    bm256n64k32_fp16<<<gridDim, blockDim>>>(a, b, c, M, N, K);
}
/*********************************************************************
 * Reduce
 *********************************************************************/
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
/**
 * @brief max reduce
 * @param g_idata: 输入数据, shape: [B, N, in_channel]
 * @param g_odata: 输出数据, shape: [B, blks, in_channel]
 * @param N
 */
template <const int blockSize = 128, const int width = 1>
__global__ void _maxReduce_f16(half *g_idata, half *g_odata, int N)
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
    constexpr int widthSize = width << 3;
    constexpr int PAD = 8;
    __shared__ half sdata[widthSize][blockSize + PAD];
    half rdata[widthSize], tmp[widthSize];
    int tid = threadIdx.y;
    size_t g_idata_planeSize = N * gridDim.x * widthSize;
    size_t g_idata_batch_base = g_idata_planeSize * blockIdx.z;

    size_t i = g_idata_batch_base + ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * widthSize;
#pragma unroll
    for (int j = 0; j < width; j++)
        HALF8(rdata[j * 8]) = HALF8(g_idata[i + j * 8]); // 8*half
    // 将不足构成下一个block的部分进行算
    {
        int line_stride = blockDim.y * gridDim.y;
        int stride = line_stride * gridDim.x * widthSize;
        int cnt = blockIdx.y * blockDim.y + tid + line_stride;
        i += stride;
        while (cnt < N)
        {
#pragma unroll
            for (int j = 0; j < width; j++)
                HALF8(tmp[j * 8]) = HALF8(g_idata[i + j * 8]);
#pragma unroll
            for (int j = 0; j < widthSize; j++)
                rdata[j] = MAX_F16(rdata[j], tmp[j]);
            cnt += line_stride;
            i += stride;
        }
    }
#pragma unroll
    for (int j = 0; j < widthSize; j++)
        sdata[j][tid] = rdata[j];
    __syncthreads();
    // blockSize*2-way reduction
    {
        if (blockSize >= 256)
        {
            if (tid < 128)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 128]);
            }
            __syncthreads();
        }
        if (blockSize >= 128)
        {
            if (tid < 64)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 64]);
            }
            __syncthreads();
        }
        if (blockSize >= 64)
        {
            if (tid < 32)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 32]);
            }
        }
        if (blockSize >= 32)
        {
            if (tid < 16)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 16]);
            }
        }
        if (blockSize >= 16)
        {
            if (tid < 8)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 8]);
            }
        }
        if (blockSize >= 8)
        {
            if (tid < 4)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 4]);
            }
        }
        if (blockSize >= 4)
        {
            if (tid < 2)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 2]);
            }
        }
        if (blockSize >= 2)
        {
            if (tid == 0)
            {
#pragma unroll
                for (int j = 0; j < widthSize; j++)
                    sdata[j][tid] = MAX_F16(sdata[j][tid], sdata[j][tid + 1]);
            }
        }
    }

    if (tid == 0)
    {
#pragma unroll
        for (int j = 0; j < widthSize; j++)
            rdata[j] = sdata[j][0];
        int g_odata_n_base = blockIdx.z * gridDim.x * gridDim.y * widthSize + (blockIdx.y * gridDim.x + blockIdx.x) * widthSize;
#pragma unroll
        for (int j = 0; j < width; j++)
            HALF8(g_odata[g_odata_n_base + j * 8]) = HALF8(rdata[j * 8]);
    }
}
/**
 * @brief 两级调用_maxReduce实现对[B, N, C]在C维度上求max
 * @attention tmp的大小需要大于[B, 2^[log2(N)] / (2*blockSize), C]
 * @attention blockSize必须为2的幂
 * @param input: shape [B, N, C]
 * @param output: shape [B, C]
 */
template <int blockSize = 128, int width = 2>
__always_inline void maxReduce_f16_call(half *input, half *tmp, half *output, int batch_size, int N, int C)
{
    dim3 blockDim(1, blockSize, 1);
    int blks = CEIL(MAX_LB_POWER2(N), blockSize);
    constexpr int widthSize = width << 3;
    dim3 gridDim(C / widthSize, blks, batch_size);
#ifdef DEBUG
    if (blockSize * width > 256 || blks > 128)
        std::cout << "Too big N or too small blockSize" << std::endl;
#endif
    if (blks > 1)
        _maxReduce_f16<blockSize, width><<<gridDim, blockDim>>>(input, tmp, N);
    else
    {
        // no up level reduce is needed
        _maxReduce_f16<blockSize, width><<<gridDim, blockDim>>>(input, output, N);
        return;
    }
    // TODO: check whether synchronization is required
    // recurse to N = blks
    dim3 blockDim_(1, blks, 1);
    dim3 gridDim_(C / widthSize, 1, batch_size);
    switch (blockDim_.y)
    {
    case 512:
        _maxReduce_f16<512, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 256:
        _maxReduce_f16<256, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 128:
        _maxReduce_f16<128, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 64:
        _maxReduce_f16<64, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 32:
        _maxReduce_f16<32, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 16:
        _maxReduce_f16<16, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 8:
        _maxReduce_f16<8, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 4:
        _maxReduce_f16<4, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    case 2:
        _maxReduce_f16<2, width><<<gridDim_, blockDim_>>>(tmp, output, blks);
        break;
    default:
        return;
    }
}
/*********************************************************************
 * Elementwise
 *********************************************************************/
template <int blockSize = 16>
__global__ void argmaxWitLabels_fp16(half *input, int *output, int *labels, int N, int dim)
{
    int tid = threadIdx.x;
    int bid = blockIdx.y;
    __shared__ half shm_val[blockSize];
    __shared__ int shm_idx[blockSize];
    shm_val[tid] = tid < dim ? input[bid * dim + tid] : __float2half(-6e4);
    shm_idx[tid] = tid;
    __syncthreads();
    half tm;
    if (tid < blockSize / 2)
    {
        tm = MAX_F16(shm_val[tid], shm_val[tid + 8]);
        shm_idx[tid] = tm == shm_val[tid + 8] ? shm_idx[tid + 8] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = MAX_F16(shm_val[tid], shm_val[tid + 4]);
        shm_idx[tid] = tm == shm_val[tid + 4] ? shm_idx[tid + 4] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = MAX_F16(shm_val[tid], shm_val[tid + 2]);
        shm_idx[tid] = tm == shm_val[tid + 2] ? shm_idx[tid + 2] : shm_idx[tid];
        shm_val[tid] = tm;
        __syncthreads();
        tm = MAX_F16(shm_val[tid], shm_val[tid + 1]);
        shm_idx[tid] = tm == shm_val[tid + 1] ? shm_idx[tid + 1] : shm_idx[tid];
    }
    output[bid] = ((shm_idx[0] == labels[bid]) ? 1 : 0);
    // output[bid] = shm_idx[0];
}
__always_inline void argmaxWitLabels_fp16_call(half *input, int *output, int *labels, int N, int dim)
{
    dim3 blockDim(16, 1);
    dim3 gridDim(1, N);
    argmaxWitLabels_fp16<16><<<gridDim, blockDim>>>(input, output, labels, N, dim);
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
    {
        perror("opendir");
    }
    return files;
}
std::string getFileNameFromPath(const std::string &path)
{
    size_t lastSlashPos = path.find_last_of("/");
    if (lastSlashPos == std::string::npos)
        return path;
    return path.substr(lastSlashPos + 1);
}

half *read_param(const std::string &filepath)
{
    // 读取 .txt 文件并转换为 std::vector<float>
    std::vector<half> data;
    std::ifstream file(filepath);
    if (file.is_open())
    {
        float value;
        while (file >> value)
        {
            data.push_back(__float2half(value));
        }
        file.close();
    }
    else
        std::cerr << "Unable to open file: " << filepath << std::endl;
    half *tmp, *ptr;
    CUDA_CHECK(cudaMalloc(&tmp, data.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(tmp, data.data(), data.size() * sizeof(half), cudaMemcpyHostToDevice));
    // transpose all fc/conv.weight
    std::string fname = getFileNameFromPath(filepath);
    if (weight_param_shape.count(fname))
    {
        int size = weight_param_shape[fname][0] * weight_param_shape[fname][1];
        if (size != data.size())
            perror("size not match");
        CUDA_CHECK(cudaMalloc(&ptr, data.size() * sizeof(half)));
        transpose2D_f16_call((half *)tmp, (half *)ptr, weight_param_shape[fname][0], weight_param_shape[fname][1]);
        CUDA_CHECK(cudaFree(tmp));
    }
    else
        ptr = tmp;
    return ptr;
}
std::unordered_map<std::string, half *> read_params(std::string dir)
{
    // std::string dir = "."; // 当前目录
    std::unordered_map<std::string, half *> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto &file : param_files)
    {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        if (filename.find("num_batches_tracked") == std::string::npos)
        {
            params[filename] = read_param(dir + "/" + file);
        }
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
void preProcessBNLayer(std::unordered_map<std::string, half *> params)
{
    for (int i = 0; i < bn_layers_size.size(); i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        half *running_mean = params[bn_layers_size[i].first + ".running_mean"];
        half *running_var = params[bn_layers_size[i].first + ".running_var"];
        half *weight = params[bn_layers_size[i].first + ".weight"];
        half *bias = params[bn_layers_size[i].first + ".bias"];
        PreprocessBN_f16_call(weight, bias, running_mean, running_var, bn_layers_size[i].second, streams[i]);
    }
    for (int i = 0; i < bn_layers_size.size(); i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
}
/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/
void pre_process_points(std::vector<float> &x, hsize_t length)
{
    x.size() > length ? x.resize(length) : x.resize(length, 0.0f);
}
using namespace H5;
int read_h5_file(const std::string &file_path, half *list_of_points, int *list_of_labels, int fix_length)
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
            dataset.read(points.data(), H5::PredType::NATIVE_FLOAT);
            // 定长处理
            pre_process_points(points, fix_length * dims[1]);
            // 存储点云数据
            // list_of_points.push_back(points);
            // memcpy(list_of_points + cnt * fix_length * 3, points.data(), fix_length * 3 * sizeof(float));
            half *dst = list_of_points + cnt * fix_length * 3;
            for (int i = 0; i < fix_length * 3; i++)
                dst[i] = __float2half(points[i]);
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
/**************************************************************************************
 * infer
 */
void batch_infer(std::unordered_map<std::string, half *> &params, half *batch_data, half *blk_out[], half *tmp, half *batch_out, int truncated_length, int batch_size)
{
    // 推理batch
    half *blk1_out = blk_out[1], *blk2_out = blk_out[2], *blk3_out = blk_out[3], *blk4_out = blk_out[4], *blk5_out = blk_out[5], *blk6_out = blk_out[6], *blk7_out = blk_out[7], *blk8_out = blk_out[8], *blk9_out = blk_out[9], *blk10_out = blk_out[10], *blk11_out = blk_out[11];

    // stn
    // check_helper(batch_data, batch_size, truncated_length, 3); // pass
    Conv1d_BN_ReLU_fp16_call(batch_data, params["feat.stn.conv1.weight"], params["feat.stn.conv1.bias"], params["feat.stn.bn1.weight"], params["feat.stn.bn1.bias"], blk1_out, batch_size, truncated_length, 3, 64); // feat.stn.conv1.weight=[64, 3], feat.stn.conv1.bias=[64], feat.stn.bn1.running_mean=[64], feat.stn.bn1.running_var=[64], feat.stn.bn1.weight=[64], feat.stn.bn1.bias=[64]

    // check_helper(blk1_out, batch_size, truncated_length, 64); // pass

    Conv1d_BN_ReLU_fp16_call(blk1_out, params["feat.stn.conv2.weight"], params["feat.stn.conv2.bias"], params["feat.stn.bn2.weight"], params["feat.stn.bn2.bias"], blk2_out, batch_size, truncated_length, 64, 128); // feat.stn.conv2.weight=[128, 64], feat.stn.conv2.bias=[128], feat.stn.bn2.running_mean=[128], feat.stn.bn2.running_var=[128], feat.stn.bn2.weight=[128], feat.stn.bn2.bias=[128]

    Conv1d_BN_ReLU_fp16_call(blk2_out, params["feat.stn.conv3.weight"], params["feat.stn.conv3.bias"], params["feat.stn.bn3.weight"], params["feat.stn.bn3.bias"], blk3_out, batch_size, truncated_length, 128, 1024); // feat.stn.conv3.weight=[1024, 128], feat.stn.conv3.bias=[1024], feat.stn.bn3.running_mean=[1024], feat.stn.bn3.running_var=[1024], feat.stn.bn3.weight=[1024], feat.stn.bn3.bias=[1024]

    // check_helper(blk3_out, batch_size, truncated_length, 1024); // pass

    maxReduce_f16_call(blk3_out, tmp, blk4_out, batch_size, truncated_length, 1024);

    // check_helper(blk4_out, batch_size, 1, 1024); // pass

    Linear_BN_ReLU_fp16_call(blk4_out, params["feat.stn.fc1.weight"], params["feat.stn.fc1.bias"], params["feat.stn.bn4.weight"], params["feat.stn.bn4.bias"], blk5_out, batch_size, 1, 1024, 512); // feat.stn.fc1.weight=[512, 1024], feat.stn.fc1.bias=[512], feat.stn.bn4.running_mean=[512], feat.stn.bn4.running_var=[512], feat.stn.bn4.weight=[512], feat.stn.bn4.bias=[512]

    // check_helper(blk5_out, batch_size, 1, 512); //

    Linear_BN_ReLU_fp16_call(blk5_out, params["feat.stn.fc2.weight"], params["feat.stn.fc2.bias"], params["feat.stn.bn5.weight"], params["feat.stn.bn5.bias"], blk6_out, batch_size, 1, 512, 256); // feat.stn.fc2.weight=[256, 512], feat.stn.fc2.bias=[256], feat.stn.bn5.running_mean=[256], feat.stn.bn5.running_var=[256], feat.stn.bn5.weight=[256], feat.stn.bn5.bias=[256]

    // check_helper(blk6_out, batch_size, 1, 256); // pass

    m64n9k256AddBasis_fp16_call(blk6_out, params["feat.stn.fc3.weight"], params["feat.stn.fc3.bias"], blk7_out, batch_size, 256, 9); // feat.stn.fc3.weight=[9, 256], feat.stn.fc3.bias=[9]

    // check_helper(blk7_out, batch_size, 1, 9);                  // pass
    // check_helper(batch_data, batch_size, truncated_length, 3); // pass
    // feat
    bm512n3k3_fp16_call(batch_data, blk7_out, blk8_out, batch_size, truncated_length, 3, 3); // [B, N, 3] x [B, (3, 3)]

    // check_helper(blk8_out, batch_size, truncated_length, 3); // pass

    Conv1d_BN_ReLU_fp16_call(blk8_out, params["feat.conv1.weight"], params["feat.conv1.bias"], params["feat.bn1.weight"], params["feat.bn1.bias"], blk9_out, batch_size, truncated_length, 3, 64); // feat.conv1.weight=[64, 3], feat.conv1.bias=[64], feat.bn1.running_mean=[64], feat.bn1.running_var=[64], feat.bn1.weight=[64], feat.bn1.bias=[64]

    // fstn
    Conv1d_BN_ReLU_fp16_call(blk9_out, params["feat.fstn.conv1.weight"], params["feat.fstn.conv1.bias"], params["feat.fstn.bn1.weight"], params["feat.fstn.bn1.bias"], blk1_out, batch_size, truncated_length, 64, 64);    // feat.fstn.conv1.weight=[64, 64], feat.fstn.conv1.bias=[64], feat.fstn.bn1.running_mean=[64], feat.fstn.bn1.running_var=[64], feat.fstn.bn1.weight=[64], feat.fstn.bn1.bias=[64]
    Conv1d_BN_ReLU_fp16_call(blk1_out, params["feat.fstn.conv2.weight"], params["feat.fstn.conv2.bias"], params["feat.fstn.bn2.weight"], params["feat.fstn.bn2.bias"], blk2_out, batch_size, truncated_length, 64, 128);   // feat.fstn.conv2.weight=[64, 64], feat.fstn.conv2.bias=[64], feat.fstn.bn2.running_mean=[64], feat.fstn.bn2.running_var=[64], feat.fstn.bn2.weight=[64], feat.fstn.bn2.bias=[64]
    Conv1d_BN_ReLU_fp16_call(blk2_out, params["feat.fstn.conv3.weight"], params["feat.fstn.conv3.bias"], params["feat.fstn.bn3.weight"], params["feat.fstn.bn3.bias"], blk3_out, batch_size, truncated_length, 128, 1024); // feat.fstn.conv3.weight=[1024, 128], feat.fstn.conv3.bias=[1024], feat.fstn.bn3.running_mean=[1024], feat.fstn.bn3.running_var=[1024], feat.fstn.bn3.weight=[1024], feat.fstn.bn3.bias=[1024]

    maxReduce_f16_call(blk3_out, tmp, blk4_out, batch_size, truncated_length, 1024);
    // check_helper(blk4_out, batch_size, 1, 1024); // pass

    Linear_BN_ReLU_fp16_call(blk4_out, params["feat.fstn.fc1.weight"], params["feat.fstn.fc1.bias"], params["feat.fstn.bn4.weight"], params["feat.fstn.bn4.bias"], blk5_out, batch_size, 1, 1024, 512); // feat.fstn.fc1.weight=[512, 1024], feat.fstn.fc1.bias=[512], feat.fstn.bn4.running_mean=[512], feat.fstn.bn4.running_var=[512], feat.fstn.bn4.weight=[512], feat.fstn.bn4.bias=[512]
    Linear_BN_ReLU_fp16_call(blk5_out, params["feat.fstn.fc2.weight"], params["feat.fstn.fc2.bias"], params["feat.fstn.bn5.weight"], params["feat.fstn.bn5.bias"], blk6_out, batch_size, 1, 512, 256);  // feat.fstn.fc2.weight=[256, 512], feat.fstn.fc2.bias=[256], feat.fstn.bn5.running_mean=[256], feat.fstn.bn5.running_var=[256], feat.fstn.bn5.weight=[256], feat.fstn.bn5.bias=[256]

    // check_helper(blk6_out, batch_size, 1, 256); // pass

    m64n4096k256AddBasis_fp16_call(blk6_out, params["feat.fstn.fc3.weight"], params["feat.fstn.fc3.bias"], blk10_out, batch_size, 256, 4096); // feat.fstn.fc3.weight=[1024, 256], feat.fstn.fc3.bias=[1024]

    // check_helper(blk10_out, batch_size, 1, 4096); // pass

    // feat
    bm256n64k32_fp16_call(blk9_out, blk10_out, blk11_out, batch_size, truncated_length, 64, 64);

    // check_helper(blk11_out, batch_size, truncated_length, 64); // pass e:0.5

    Conv1d_BN_ReLU_fp16_call(blk11_out, params["feat.conv2.weight"], params["feat.conv2.bias"], params["feat.bn2.weight"], params["feat.bn2.bias"], blk2_out, batch_size, truncated_length, 64, 128);  // feat.conv2.weight=[128, 64], feat.conv2.bias=[128], feat.bn2.running_mean=[128], feat.bn2.running_var=[128], feat.bn2.weight=[128], feat.bn2.bias=[128]
    m256n128k32_bn_fp16_call(blk2_out, params["feat.conv3.weight"], params["feat.conv3.bias"], params["feat.bn3.weight"], params["feat.bn3.bias"], blk3_out, batch_size, truncated_length, 128, 1024); // feat.conv3.weight=[1024, 128], feat.conv3.bias=[1024], feat.bn3.running_mean=[1024], feat.bn3.running_var=[1024], feat.bn3.weight=[1024], feat.bn3.bias=[1024]

    maxReduce_f16_call(blk3_out, tmp, blk4_out, batch_size, truncated_length, 1024);

    // check_helper(blk4_out, batch_size, 1, 1024); // pass e:0.1

    // model
    Conv1d_BN_ReLU_fp16_call(blk4_out, params["fc1.weight"], params["fc1.bias"], params["bn1.weight"], params["bn1.bias"], blk5_out, batch_size, 1, 1024, 512); // fc1.weight=[512, 1024], fc1.bias=[512], bn1.running_mean=[512], bn1.running_var=[512], bn1.weight=[512], bn1.bias=[512]

    Conv1d_BN_ReLU_fp16_call(blk5_out, params["fc2.weight"], params["fc2.bias"], params["bn2.weight"], params["bn2.bias"], blk6_out, batch_size, 1, 512, 256); // fc2.weight=[256, 512], fc2.bias=[256], bn2.running_mean=[256], bn2.running_var=[256], bn2.weight=[256], bn2.bias=[256]

    GemmWithBias_fp16_call(blk6_out, params["fc3.weight"], params["fc3.bias"], batch_out, batch_size, 256, 10); // fc3.weight=[10, 256], fc3.bias=[10]

    // check_helper(blk12_out, batch_size, 1, 10); // pass
}
void cudaFreeAll(float *blk_out[], int blk_num)
{
    for (int i = 1; i < blk_num; i++)
        CUDA_CHECK(cudaFree(blk_out[i]));
    for (int i = 0; i < 15; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
}
void cuda_help()
{
    size_t avail(0); // 可用显存
    size_t total(0); // 总显存
    cudaMemGetInfo(&avail, &total);
    printf("Avaliable Memery = %dm   Usage Memory = %dm\n", int(avail / 1024 / 1024), int(total / 1024 / 1024) - int(avail / 1024 / 1024));
}
int main(int argc, char *argv[])
{
    // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
#ifdef DEBUG
    std::string dir = "/home/gpu_course/8192"; // argv[1];
#endif
#ifndef DEBUG
    std::string dir = argv[1];
#endif
    // 读取模型参数
    set_weight_param_shape();
    set_bn_layers_size();
    auto params = read_params(dir);
    preProcessBNLayer(params);
    // 读取训练集数据
#ifdef DEBUG
    std::string file_path = "/home/gpu_course/data/test_point_clouds.h5";
    // FILE *filep = freopen("/home/gpu_course/out/cu_out.txt", "w", stdout);
#endif
#ifndef DEBUG
    std::string file_path = "./data/test_point_clouds.h5";
#endif
    int truncated_length = 8192, dataset_len = 1000;
    half *list_of_points = (half *)malloc((size_t)dataset_len * truncated_length * 3 * sizeof(half)); // 93.75 MB
    int *list_of_labels = (int *)malloc((size_t)dataset_len * sizeof(int));                           // 4K
    if (list_of_labels == NULL || list_of_points == NULL)
    {
        std::cerr << "malloc failed" << std::endl;
        return -1;
    }
    dataset_len = read_h5_file(file_path, list_of_points, list_of_labels, truncated_length);
    /********************************************************************/
    int alinged_dataset_len = 1024, batch_size = 1024;
    half *batch_data;
    int *dataset_lables;
    CUDA_CHECK(cudaMalloc(&batch_data, (size_t)alinged_dataset_len * truncated_length * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dataset_lables, (size_t)dataset_len * sizeof(int))); // 1024
    // TODO some blk_out may be reused
    half *blk1_out, *blk2_out, *blk3_out, *blk4_out, *blk5_out, *blk6_out, *blk7_out, *blk8_out, *blk9_out, *blk10_out, *blk11_out, *tmp, *infer_out;
    int *infer_labels, *corrects;
    // stn
    CUDA_CHECK(cudaMalloc(&blk1_out, (size_t)batch_size * truncated_length * 64 * sizeof(half)));   // relu(bn1d(conv1d(3,64)))
    CUDA_CHECK(cudaMalloc(&blk2_out, (size_t)batch_size * truncated_length * 128 * sizeof(half)));  // relu(bn1d(conv1d(64,128)))
    CUDA_CHECK(cudaMalloc(&blk3_out, (size_t)batch_size * truncated_length * 1024 * sizeof(half))); // relu(bn1d(conv1d(128,1024)))
    CUDA_CHECK(cudaMalloc(&blk4_out, (size_t)batch_size * 1024 * sizeof(half)));                    // max(dim N)
    int blks = CEIL(MAX_LB_POWER2(truncated_length), 128);                                          // max reuduce blockNum
    CUDA_CHECK(cudaMalloc(&tmp, (size_t)batch_size * blks * 1024 * sizeof(half)));                  // max_reduce_tmp, blockSize=128
    CUDA_CHECK(cudaMalloc(&blk5_out, (size_t)batch_size * 512 * sizeof(half)));                     // mlp(1024,512)
    CUDA_CHECK(cudaMalloc(&blk6_out, (size_t)batch_size * 256 * sizeof(half)));                     // mlp(512,256)
    CUDA_CHECK(cudaMalloc(&blk7_out, (size_t)batch_size * 9 * sizeof(half)));                       // mlp(256,9)+add(9)
    // feat
    CUDA_CHECK(cudaMalloc(&blk8_out, (size_t)batch_size * truncated_length * 3 * sizeof(half)));   // bmm(3,3)
    CUDA_CHECK(cudaMalloc(&blk9_out, (size_t)batch_size * truncated_length * 64 * sizeof(half)));  // relu(bn1d(conv1d(3,64)))
    CUDA_CHECK(cudaMalloc(&blk11_out, (size_t)batch_size * truncated_length * 64 * sizeof(half))); // bmm(64,64)
    // fstn reuse stn
    CUDA_CHECK(cudaMalloc(&blk10_out, (size_t)batch_size * 64 * 64 * sizeof(half))); // mlp(256,4096)+add(4096)
    // model
    CUDA_CHECK(cudaMalloc(&infer_out, (size_t)alinged_dataset_len * 10 * sizeof(half))); // logsoftmax(10)
    CUDA_CHECK(cudaMalloc(&infer_labels, alinged_dataset_len * sizeof(int)));
    CUDA_CHECK(cudaMemset(infer_labels + dataset_len, 0, (size_t)(alinged_dataset_len - dataset_len) * sizeof(int)));
    half *blk_outs[] = {NULL, blk1_out, blk2_out, blk3_out, blk4_out, blk5_out, blk6_out, blk7_out, blk8_out, blk9_out, blk10_out, blk11_out};
    // 188M
    CUDA_CHECK(cudaMemcpy(batch_data, list_of_points, (size_t)dataset_len * truncated_length * 3 * sizeof(half), cudaMemcpyHostToDevice));
    // 4K
    CUDA_CHECK(cudaMemcpy(dataset_lables, list_of_labels, (size_t)dataset_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocManaged(&corrects, sizeof(int)));
    corrects[0] = 0;
    /********************************************************************/
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    batch_infer(params, batch_data, blk_outs, tmp, infer_out, truncated_length, batch_size);
    argmaxWitLabels_fp16_call(infer_out, infer_labels, dataset_lables, dataset_len, 10);
    sumReduce_call(infer_labels, corrects, dataset_len);
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    CUDA_CHECK(cudaGetLastError());
    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << (float)(corrects[0]) / (float)dataset_len;
    return 0;
}
// #endif