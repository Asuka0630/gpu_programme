/*
    m128n128k8 is a signle-precision gemm kernel, without tensor core can be used capability >= 6.1
        · achieve 85%+ of fp32 peak performance
        · bounded by 73%+ of memory performance

    m256n64k32 is a half-precision gemm kernel, with tensor core can be used capability >= 7.0
        [accmulator registers layout is V100 style]

    m128n128k32 is a half-precision gemm kernel, with tensor core can be used capability >= 9.0
        [accmulator registers layout is H100 style]
        · memory bounded

    device: NVIDIA H100 PCIe
    results:
        kernel: m128n128k8
        SGEMM [2048 2048 2048]: latency: 0.66 ms
        SGEMM [4096 4096 4096]: latency: 3.75 ms
        SGEMM [8192 8192 8192]: latency: 29.42 ms
        SGEMM [16384 16384 16384]: latency: 235.28 ms

        kernel: m128n128k32
        HGEMM [2048 2048 2048] latency: : 0.23 ms
        HGEMM [4096 4096 4096] latency: : 1.94 ms
        HGEMM [8192 8192 8192] latency: : 13.88 ms
        HGEMM [16384 16384 16384] latency: : 106.32 ms

    TODO:
        1. TMA
        2. async load
*/
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <vector>
#include <time.h>
#include <math_constants.h>
#include "assert.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define CDIV(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// a vector of 4 floats
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define HALF8(pointer) FLOAT4(pointer)

#define CUDA_CHECK(Err) _cuda_check(Err, __FILE__, __LINE__)
void _cuda_check(cudaError_t err, char const *file, int line)
{
    do
    {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error at %s %d: %s\n",
                    __FILE__, __LINE__, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    } while (0);
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const *file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**************************************************************/
__global__ void m128n128k8(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, float *__restrict__ v, const size_t M, const size_t N, const size_t K)
{
    // C = AB + v
    // A is M*K
    // B is K*N
    // each thread block solve a 128*128 block of C
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    // without CTA swizzle
    // const int bx = blockIdx.x;
    // with CTA swizzle
    const int bx = blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    // blockDim is 2D, contains 256 threads
    // without warp swizzle
    // const int tx = threadIdx.x, ty = threadIdx.y;
    // const int tid = ty * blockDim.x + tx;
    // blockDim is 1D, contains 256 threads
    const int tid = threadIdx.x;
    // warp swizzle
    // the layout of threads in a warp 2*16 -> 4*8
    /*
        each thread solve 8x8 sub tile
        0                         63|64                        127
        +———————————————————————————+———————————————————————————+ 0
        |T0|T1|...|T7 |T128|...|T135|                           |
        |T8|T9|...|T15|     ...     |                           |
        |     ...     |     ...     |                           |
        |     ...     |     ...     |                           |
        |T120|...|T127|T248|...|T255|                           |
        +———————————————————————————+———————————————————————————+
        |                           |                           |
        |                           |                           |
        |                           |                           |
        |                           |                           |
        |                           |                           |
        +———————————————————————————+———————————————————————————+ 127
     */
    const int tx = (tid & 7) + (tid >= 128 ? 8 : 0); // tid % 8 + (tid / 128) * 8;
    const int ty = (tid >> 3) & 15;                  //(tid / 8) % TM;
    __shared__ float s_a[2][BK][BM];                 // avoid conflict when load to register
    __shared__ float s_b[2][BK][BN];
    __shared__ float s_v[BN];

    float r_load_a[4];
    float r_load_b[4];

    // use to load data from shared memory
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0}; // each thread solve a 8*8 block of C

    const int pack_size = 4; // FLOAT4
    // assert(K % pack_size == 0 and N % pack_size == 0);
    // each thread should load 4B(128*8/256) of A
    size_t load_a_smem_m = tid / (BK / pack_size); // BK/pack_size threads is need to load one line of A
    size_t load_a_smem_k = (tid & 1) * pack_size;  // (tid % (BK/pack_size)) * pack_size
    // each thread should load 4B(128*8/256) of B
    size_t load_b_smem_k = tid / (BN / pack_size); // BN/pack_size threads is need to load one line of B
    size_t load_b_smem_n = (tid & 31) * pack_size; // (tid % (BN/pack_size)) * pack_size

    size_t load_a_gmem_m = by * BM + load_a_smem_m;
    size_t load_b_gmem_n = bx * BN + load_b_smem_n;

    float4 z4 = {0, 0, 0, 0};
    // Double buffering
    {
        size_t load_a_gmem_k = load_a_smem_k;
        size_t load_a_gmem_off = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        size_t load_b_gmem_k = load_b_smem_k;
        size_t load_b_gmem_off = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        if (load_a_gmem_m < M && load_a_gmem_k < K)
            FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_off]);
        else
            FLOAT4(r_load_a[0]) = z4;
        if (load_b_gmem_n < N && load_b_gmem_k < K)
        {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_b_gmem_off]);
            // TODO: use cooperative_groups::memcpy_async
        }
        else
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = z4;
        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        if (tid < (BN / pack_size))
        {
            if (load_b_gmem_n + pack_size <= N)
            {
                FLOAT4(s_v[load_b_smem_n]) = FLOAT4(v[load_b_gmem_n]);
                // TODO: use cooperative_groups::memcpy_async
            }
            else
                FLOAT4(s_v[load_b_smem_n]) = z4;
        }
        __syncthreads();
    }

    int smem_sel_next;
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {
        // next block
        smem_sel_next = bk & 1;
        // last block
        int smem_sel = (bk - 1) & 1;

        size_t load_a_gmem_k = bk * BK + load_a_smem_k;
        size_t load_a_gmem_off = OFFSET(load_a_gmem_m, load_a_gmem_k, K);

        size_t load_b_gmem_k = bk * BK + load_b_smem_k;
        size_t load_b_gmem_off = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // load next block
        {
            if (load_a_gmem_m < M && load_a_gmem_k < K)
                FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_off]);
            else
                FLOAT4(r_load_a[0]) = z4;
            if (load_b_gmem_n < N && load_b_gmem_k < K)
                FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_off]);
            else
                FLOAT4(r_load_b[0]) = z4;
        }

        // compute last block
        {
#pragma unroll
            for (int tk = 0; tk < BK; tk++)
            {
                // load last from shared memory
                FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
                FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
                FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
                FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
                // why register tiling
                /*
                                +— 8—+
                                +———————————————+
                                |    |          |
                                +———————————————+
                    +   +————+  +———————————————+
                    8   |    |  | 8x8|          |
                    +   |————|  |————+          |
                        |    |  |               |
                        |    |  |               |
                        |    |  |               |
                        +————+  +———————————————+

                    bank conflict theory:
                        1. 如果warp中每个thread访问4Byte  [LDS.32],  请求不会被拆分,  需要在每1/1个warp内判断是否存在conflict
                        2. 如果warp中每个thread访问8Byte  [LDS.64],  请求会被拆成2次, 需要在每1/2个warp内判断是否存在conflict
                        3. 如果warp中每个thread访问16Byte [LDS.128], 请求会被拆成4次, 需要在每1/4个warp内判断是否存在conflict

                    为尽量减少指令数量, 考虑LDS.128访问shared memory
                    在BM=BN=128的情况下, BM/TM=BN/TN=16, 也就是计算BM*BN的子块的一行需要1/2warp

                    如果layout of warp是2*16
                       [[0, 1, 2, ... ,15]
                        [16,17,18,... ,31]]
                       依赖B中8*8*4=64*4=256B>128B的数据, 同一时刻1/4warp的线程访问B时存在bank conflict
                    依赖的B数据两分成两次读取 对于第一个warp的1/4个warp依赖的B中的数据地址为
                    tid: 0      1       2        ...
                    off: [0,3], [8,11], [16,19], ...
                    只用了1/2的带宽

                    出现这个问题的原因是, 两个线程访问的数据不是连续的, 中间间隔了4Byte的数据在下一周期访问
                    因此可以调整每个线程的计算范围, 将每个线程的计算范围拆分成两部分 两个4*4的部分
                    依赖的B数据两分成两次读取 对于第一个warp的1/4个warp依赖的B中的数据地址为
                    tid: 0      1       2      ...
                    off: [0,3], [4,7], [8,11], ...

                                +4+     +4+
                                +———————————————+
                                | |     | |     |
                                +———————————————+
                    +   +————+  +———————————————+
                    +4  |————|  |—+-----+—+     |
                        |    |  | |     | |     |
                    +   |————|  |—+-----+—+     |
                    +4  |————|  |—+-----+—+     |
                        |    |  |               |
                        +————+  +———————————————+
                */
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
    }

    // epilogue
    {
#pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            // load last from shared memory
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2 + BN / 2]);
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
    }

    // fuse bias
    {
#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                r_c[tm][tn] += s_v[(tx * TN / 2) + (tn & 3) + (tn < 4 ? 0 : BN / 2)];
            }
        }
    }

    // store
    {
#pragma unroll
        for (int i = 0; i < TM / 2; i++)
        {
            size_t store_c_gmem_m = by * BM + ty * TM / 2 + i;
            size_t store_c_gmem_n = bx * BN + tx * TN / 2;
            size_t store_c_gmem_off = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            if (store_c_gmem_m < M && store_c_gmem_n < N) // must not use store_c_gmem_off to check
                FLOAT4(C[store_c_gmem_off]) = FLOAT4(r_c[i][0]);
            if (store_c_gmem_m < M && store_c_gmem_n < N - BN / 2)
                FLOAT4(C[store_c_gmem_off + BN / 2]) = FLOAT4(r_c[i][4]);
        }
#pragma unroll
        for (int i = 0; i < TM / 2; i++)
        {
            size_t store_c_gmem_m = by * BM + ty * TM / 2 + i + BM / 2;
            size_t store_c_gmem_n = bx * BN + tx * TN / 2;
            size_t store_c_gmem_off = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            if (store_c_gmem_m < M && store_c_gmem_n < N)
                FLOAT4(C[store_c_gmem_off]) = FLOAT4(r_c[i + TM / 2][0]);
            if (store_c_gmem_m < M && store_c_gmem_n < N - BN / 2)
                FLOAT4(C[store_c_gmem_off + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        }
    }
}
template <typename T>
void launch_m128n128k8(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, float *__restrict__ v, const size_t M, const size_t N, const size_t K, cudaStream_t stream)
{
    const size_t BM = 128, BN = 128, TM = 8, TN = 8;
    const size_t pack_size = 4; // FLOAT4
    assert(K % pack_size == 0 && N % pack_size == 0 && "FLOAT4 is used");
    // without CTA swizzle
    // dim3 blockDim{BN / TN, BM / TM};
    // dim3 gridDim{static_cast<unsigned int>(CDIV(N, BN)), static_cast<unsigned int>(CDIV(M, BM))};
    // with CTA swizzle
    const size_t NSPLIT = 2048; // this is hyperparameter could be smaller when M,N,K is small
    const size_t split_num = CDIV(N, NSPLIT);
    // without warp swizzle
    // dim3 blockDim{BN / TN, BM / TM};
    // with warp swizzle
    dim3 blockDim{BN / TN * BM / TM};
    dim3 gridDim{static_cast<unsigned int>(CDIV(CDIV(N, BN), split_num)), static_cast<unsigned int>(CDIV(M, BM)), static_cast<unsigned int>(split_num)};
    m128n128k8<<<gridDim, blockDim, 0, stream>>>(A, B, C, v, M, N, K);
}

using T = half;
__global__ void m128n128k32(T *__restrict__ A, T *__restrict__ B, T *__restrict__ C, T *__restrict__ v, const size_t M, const size_t N, const size_t K)
{
    // This kernel can be used to solve C = AB +v
    /*
        to solve 128*128 block, each thread group contains 128/32 = 4 warps
        each warp solve a 64*64 block of C, we use m16n16k16 tensor core
        the layout of warps in a CTA is like
        +——  64  ——+——  64  ——+
        +——————————+——————————+ +
        |          |          | |
        |    0     |    1     | 64
        |          |          | |
        +——————————+——————————+ +
        |          |          | |
        |    2     |    3     | 64
        |          |          | |
        +——————————+——————————+ +
     */
    const int BM = 128, BN = 128, BK = 32;
    // const size_t bx = blockIdx.x;
    const size_t bx = blockIdx.z * gridDim.x + blockIdx.x;
    const size_t by = blockIdx.y;
    // 1D thread block, contains 128 threads
    const size_t tid = threadIdx.x;
    const size_t warp_id = tid / 32;
    const int lane_id = tid & 31;
    // using padding trick to alleviate bank conflict
    const int PADDING = 8;
    const size_t PADDED_BK = BK + PADDING;
    const size_t PADDED_BN = BN + PADDING;
    __shared__ T s_a[2][BM][PADDED_BK];
    __shared__ T s_b[2][BK][PADDED_BN];
    __shared__ T s_v[BN];
    /*
        within each iteration
        each thread load 64x32 block of A and 32*64 block of B
        frag_a[2][4]:
            +— 16 —+
            +——————+——————+ +
            |      |      | 16
            |      |      | |
            +——————+——————+ +
            |      |      |
            |      |      |
            +——————+——————+
            |      |      |
            |      |      |
            +——————+——————+
            |      |      |
            |      |      |
            +——————+——————+
    */
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> frag_a[2][4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major> frag_b[2][4];
    // for frag_c is shaped like 64*64 so it requires 4*4 fragments
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T> frag_c[4][4];
    // fill frag_c 0.0
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
    }

    // use HALF8(16B, LDS.128) vectorized memory access
    // to load BM*BK from A in each iteration, each thread must load BM*BK/128/8 = 4 HALF8
    // assert (BK % 8 == 0 && "must be HALF8 aligned")
    const int pack_size = 8;
    /*
        how to load block of A
        +————    32    —————+
        +———————————————————+ +   +
        | T0 | T1 | T2 | T3 | |   |
        | T4 | T5 | T6 | T7 | 32  |
        |        ...        |     |
        |T124|T125|T126|T127| |
        +———————————————————+ +
        |        ...        |    128
        +———————————————————+
        | T0 | T1 | T2 | T3 |
        | T4 | T5 | T6 | T7 |     |
        |        ...        |     |
        |T124|T125|T126|T127|     |
        +———————————————————+     +
    */
    // BK/8=4 threads is required to load one line from block A
    const size_t load_a_smem_m = tid / (BK / pack_size);
    // stride of load a in m direction is 128/4 = 32
    const size_t load_a_smem_stride_m = blockDim.x / (BK / pack_size);
    const size_t load_a_smem_k = (tid % (BK / pack_size)) * pack_size;

    // to load BK*BN from B in each iteration, each thread must load BK*BN/128/8 = 4 HALF8
    // assert (BN % 8 == 0 && "must be HALF8 aligned")
    // BN/8=16 threads is required to load one line from block B
    const size_t load_b_smem_k = tid / (BN / pack_size);
    const size_t load_b_smem_stride_k = blockDim.x / (BN / pack_size);
    const size_t load_b_smem_n = (tid % (BN / pack_size)) * pack_size;

    const size_t load_a_gmem_m = by * BM + load_a_smem_m;
    const size_t load_b_gmem_n = bx * BN + load_b_smem_n;

    // load bias
    if (tid < (BN / pack_size)) // BN*2B=256B/128=2 bursts
    {
        if (load_b_smem_n + pack_size <= N)
            HALF8(s_v[load_b_smem_n]) = HALF8(v[load_b_gmem_n]);
    }
    int sel_smem_next = 0;
    size_t load_a_gmem_k = load_a_smem_k;
    size_t load_b_gmem_k = load_b_smem_k;

    // prologue
    // prefetch
    {
        size_t load_a_smem_m_cur = load_a_smem_m;
        size_t load_a_gmem_m_cur = load_a_gmem_m;
        while (load_a_smem_m_cur < BM && load_a_gmem_m_cur < M)
        {
            if (load_a_gmem_k + pack_size <= K)
                HALF8(s_a[sel_smem_next][load_a_smem_m_cur][load_a_smem_k]) = HALF8(A[OFFSET(load_a_gmem_m_cur, load_a_gmem_k, K)]);
            load_a_smem_m_cur += load_a_smem_stride_m;
            load_a_gmem_m_cur += load_a_smem_stride_m;
        }
        size_t load_b_smem_k_cur = load_b_smem_k;
        size_t load_b_gmem_k_cur = load_b_gmem_k;
        while (load_b_smem_k_cur < BK && load_b_gmem_k_cur < K)
        {
            if (load_b_gmem_n + pack_size <= N)
                HALF8(s_b[sel_smem_next][load_b_smem_k_cur][load_b_smem_n]) = HALF8(B[OFFSET(load_b_gmem_k_cur, load_b_gmem_n, N)]);
            load_b_smem_k_cur += load_b_smem_stride_k;
            load_b_gmem_k_cur += load_b_smem_stride_k;
        }
        __syncthreads();
    }
    // layout of warps are 2*2
    size_t comp_c_frag_m = warp_id >> 1;
    size_t comp_c_frag_n = warp_id & 1;
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {
        int sel_smem = (bk - 1) & 1;
        sel_smem_next = bk & 1;

        load_a_gmem_k += BK;
        load_b_gmem_k += BK;

        // solve last block
        {
            // move from share memory to frag_a
            nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[sel_smem][comp_c_frag_m * 64][0], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[sel_smem][comp_c_frag_m * 64 + 16][0], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[sel_smem][comp_c_frag_m * 64 + 32][0], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[sel_smem][comp_c_frag_m * 64 + 48][0], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[sel_smem][comp_c_frag_m * 64][16], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[sel_smem][comp_c_frag_m * 64 + 16][16], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[sel_smem][comp_c_frag_m * 64 + 32][16], PADDED_BK);
            nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[sel_smem][comp_c_frag_m * 64 + 48][16], PADDED_BK);
            // move from share memory to frag_b
            nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[sel_smem][0][comp_c_frag_n * 64], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[sel_smem][0][comp_c_frag_n * 64 + 16], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[sel_smem][0][comp_c_frag_n * 64 + 32], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[sel_smem][0][comp_c_frag_n * 64 + 48], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[sel_smem][16][comp_c_frag_n * 64], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[sel_smem][16][comp_c_frag_n * 64 + 16], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[sel_smem][16][comp_c_frag_n * 64 + 32], PADDED_BN);
            nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[sel_smem][16][comp_c_frag_n * 64 + 48], PADDED_BN);
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
        }

        // load next block
        {
            size_t load_a_smem_m_cur = load_a_smem_m;
            size_t load_a_gmem_m_cur = load_a_gmem_m;
            while (load_a_smem_m_cur < BM && load_a_gmem_m_cur < M)
            {
                if (load_a_gmem_k + pack_size <= K)
                    HALF8(s_a[sel_smem_next][load_a_smem_m_cur][load_a_smem_k]) = HALF8(A[OFFSET(load_a_gmem_m_cur, load_a_gmem_k, K)]);
                load_a_smem_m_cur += load_a_smem_stride_m;
                load_a_gmem_m_cur += load_a_smem_stride_m;
            }
            size_t load_b_smem_k_cur = load_b_smem_k;
            size_t load_b_gmem_k_cur = load_b_gmem_k;
            while (load_b_smem_k_cur < BK && load_b_gmem_k_cur < K)
            {
                if (load_b_gmem_n + pack_size <= N)
                    HALF8(s_b[sel_smem_next][load_b_smem_k_cur][load_b_smem_n]) = HALF8(B[OFFSET(load_b_gmem_k_cur, load_b_gmem_n, N)]);
                load_b_smem_k_cur += load_b_smem_stride_k;
                load_b_gmem_k_cur += load_b_smem_stride_k;
            }
            __syncthreads();
        }
    }

    // epilogue
    {
        // move from share memory to frag_a
        nvcuda::wmma::load_matrix_sync(frag_a[0][0], &s_a[sel_smem_next][comp_c_frag_m * 64][0], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[0][1], &s_a[sel_smem_next][comp_c_frag_m * 64 + 16][0], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[0][2], &s_a[sel_smem_next][comp_c_frag_m * 64 + 32][0], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[0][3], &s_a[sel_smem_next][comp_c_frag_m * 64 + 48][0], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[1][0], &s_a[sel_smem_next][comp_c_frag_m * 64][16], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[1][1], &s_a[sel_smem_next][comp_c_frag_m * 64 + 16][16], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[1][2], &s_a[sel_smem_next][comp_c_frag_m * 64 + 32][16], PADDED_BK);
        nvcuda::wmma::load_matrix_sync(frag_a[1][3], &s_a[sel_smem_next][comp_c_frag_m * 64 + 48][16], PADDED_BK);
        // move from share memory to frag_b
        nvcuda::wmma::load_matrix_sync(frag_b[0][0], &s_b[sel_smem_next][0][comp_c_frag_n * 64], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[0][1], &s_b[sel_smem_next][0][comp_c_frag_n * 64 + 16], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[0][2], &s_b[sel_smem_next][0][comp_c_frag_n * 64 + 32], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[0][3], &s_b[sel_smem_next][0][comp_c_frag_n * 64 + 48], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[1][0], &s_b[sel_smem_next][16][comp_c_frag_n * 64], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[1][1], &s_b[sel_smem_next][16][comp_c_frag_n * 64 + 16], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[1][2], &s_b[sel_smem_next][16][comp_c_frag_n * 64 + 32], PADDED_BN);
        nvcuda::wmma::load_matrix_sync(frag_b[1][3], &s_b[sel_smem_next][16][comp_c_frag_n * 64 + 48], PADDED_BN);
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
    }

    // fuse
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                // frag_c[i][j] consists a group of registers from threads in a warp
                // frag_c[i][j] is a 16*16 block so frag_c[i][j].num_elements = 16*16/32 = 8
                for (int t = 0; t < frag_c[i][j].num_elements; t++)
                {
                    /*
                        the layout of frag_c[4][4] in H100
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
                        |               \
                        |                ——————————————————
                        |   the layout of frag_c in H100   \
                        +——————                    16                      —————+
                        +————————     8      ———————+——————       8        —————+
                        +—  2 —+—  2 —+—    ...   ——+
                        +———————————————————————————————————————————————————————+ + + +
                        |T0:0-1|  T1  |  T2  |  T3  |T0:4-5|  T1  |  T2  |  T3  | 1 | |
                        |  T4  |  T5  |  T6  |  T7  |  T4  |  T5  |  T6  |  T7  | + | |
                        |  T8  |  T9  |  T10 |  T11 |  T8  |  T9  |  T10 |  T11 |   | |
                        |  T12 |  T13 |  T14 |  T15 |  T12 |  T13 |  T14 |  T15 |   8 |
                        |  T16 |  T17 |  T18 |  T19 |  T16 |  T17 |  T18 |  T19 |     |
                        |  T20 |  T21 |  T22 |  T23 |  T20 |  T21 |  T22 |  T23 |   | |
                        |  T24 |  T25 |  T26 |  T27 |  T24 |  T25 |  T26 |  T27 |   |
                        |  T28 |  T29 |  T30 |  T31 |  T28 |  T29 |  T30 |  T31 |   |
                        +---------------------------+---------------------------+   + 16
                        |T0:2-3|  T1  |  T2  |  T3  |T0:6-7|  T1  |  T2  |  T3  |
                        |  T4  |  T5  |  T6  |  T7  |  T4  |  T5  |  T6  |  T7  |
                        |  T8  |  T9  |  T10 |  T11 |  T8  |  T9  |  T10 |  T11 |     |
                        |  T12 |  T13 |  T14 |  T15 |  T12 |  T13 |  T14 |  T15 |     |
                        |  T16 |  T17 |  T18 |  T19 |  T16 |  T17 |  T18 |  T19 |     |
                        |  T20 |  T21 |  T22 |  T23 |  T20 |  T21 |  T22 |  T23 |     |
                        |  T24 |  T25 |  T26 |  T27 |  T24 |  T25 |  T26 |  T27 |     |
                        |  T28 |  T29 |  T30 |  T31 |  T28 |  T29 |  T30 |  T31 |     |
                        +———————————————————————————————————————————————————————+     +
                        */
                    // you can fuse any operator here
                    int col = comp_c_frag_n * 64 + j * 16 + (t / 4) * 8 + (t & 1) + (lane_id & 3) * 2;
                    float tmp = static_cast<float>(frag_c[i][j].x[t]) + static_cast<float>(s_v[col]); // fuse bias
                    // tmp = tmp > 0.0f ? tmp : 0.0f;                                                    // fuse relu
                    frag_c[i][j].x[t] = static_cast<T>(tmp);
                }
            }
        }
    }
    // store
    {
        const size_t store_c_gmem_m = by * BM + comp_c_frag_m * 64;
        const size_t store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                if (store_c_gmem_m + i * 16 < M && store_c_gmem_n + j * 16 < N)
                    nvcuda::wmma::store_matrix_sync(&C[OFFSET(store_c_gmem_m + i * 16, store_c_gmem_n + j * 16, N)],
                                                    frag_c[i][j], N, nvcuda::wmma::mem_row_major);
            }
        }
    }
}
template <typename T>
void launch_m128n128k32(T *__restrict__ A, T *__restrict__ B, T *__restrict__ C, T *__restrict__ v, const size_t M, const size_t N, const size_t K, cudaStream_t stream)
{
    const size_t BM = 128, BN = 128;
    const size_t pack_size = 8; // HALF8
    assert(K % pack_size == 0 && N % pack_size == 0 && "HALF8 is used");
    dim3 blockDim{128};
    // dim3 gridDim{static_cast<unsigned int>(CDIV(N, BN)), static_cast<unsigned int>(CDIV(M, BM))};
    const size_t NSPLIT = 2048; // this is hyperparameter could be smaller when M,N,K is small
    const size_t split_num = CDIV(N, NSPLIT);
    dim3 gridDim{static_cast<unsigned int>(CDIV(CDIV(N, BN), split_num)), static_cast<unsigned int>(CDIV(M, BM)), static_cast<unsigned int>(split_num)};
    m128n128k32<<<gridDim, blockDim, 0, stream>>>(A, B, C, v, M, N, K);
}

__global__ void m256n64k32(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c, half *__restrict__ v, const int M, const int N, const int K)
{
    // CTA一共128个线程, 128/32 = 4个warp
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

    // frag_a[2][4]
    // 每个线程完成BM/4xBK与BKxBN的子块得到64x64的frag_c
    // 其中[2]由于BK=32, 需要32/16=2个m16n16k16的frag_a
    // 其中[4]由于BM/4=64, 需要64/16=4个m16n16k16的frag_a
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
        对于a, 需要加载BM*BK个元素, 每个线程需要加载BM*BK/128/8=8个HALF8, 每个线程在连续的8行中加载
        对于b, 需要加载BK*BN个元素, 每个线程需要加载BK*BN/128/8=2个HALF8

        一个CTA中4个warp的分工
        comp_c_frag_m = warp_id
        comp_c_frag_n = 0
        +——  64  ——+
        +——————————+ +
        |          | |
        |    0     | 64
        |          | |
        +——————————+ +
        |          |
        |    1     |
        |          |
        +——————————+
        |          |
        |    2     |
        |          |
        +——————————+
        |          |
        |    3     |
        |          |
        +——————————+
     */
    // BK=32, 每个线程加载8个half8，因此加载BM*BK中的一行需要4个线程
    int load_a_smem_m = (tid >> 2) << 3; // tid/4*8
    int load_a_smem_k = (tid & 3) << 3;  // tid%4*8
    // BN=64, 每个线程加载8个half8，因此加载BK*BN中的一行需要8个线程
    int load_b_smem_k = (tid >> 3) << 1; // tid/8*2
    int load_b_smem_n = (tid & 7) << 3;  // tid%8*8

    // 当前CTA计算所依赖的BM*BK的行offset
    int load_a_gmem_m = by * BM + load_a_smem_m;
    // 当前CTA计算所依赖的BK*BN的列offset
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid;
    int comp_c_frag_n = 0; //

    if (tid < (BN / 8))
    {
        int load_v_gmem_addr = bx * BN + load_b_smem_n;
        HALF8(s_v[load_b_smem_n]) = HALF8(v[load_v_gmem_addr]);
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

    /*
        the layout of frag_c[4][4] V100
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
        |               \
        |                ——————————————————
        |       the layout of frag_c       \
        +——————          16            —————+
        +——————  8   —————+——————  8   —————+
        +———————————————————————————————————+ + + +
        |        0        |        8        | 1 | |
        |        1        |        9        | + 4 |
        |        2        |       10        |     |
        |        3        |       11        |   | |
        +-----------------------------------+   + |
        |       16        |       24        |     |
        |       17        |       25        |     |
        |       18        |       26        |
        |       19        |       27        |
        +-----------------------------------+     16
        |        4        |       12        |
        |        5        |       13        |
        |        6        |       14        |     |
        |        7        |       15        |     |
        +-----------------------------------+     |
        |       20        |       28        |     |
        |       21        |       29        |     |
        |       22        |       30        |     |
        |       23        |       31        |     |
        +-----------------------------------+     +
    */

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // frag_c[i][j].num_elements = 8
            // 这是因为frag_c[i][j]是一个16x16的结果, 而16*16/32=8
            // 每个线程持有8个结果寄存器
            for (int t = 0; t < frag_c[i][j].num_elements; t++)
            {
                int col = comp_c_frag_n * 64 + j * 16 + t + ((laneid / 8) & 1) * 8;
                float tmp = __half2float(frag_c[i][j].x[t]) + __half2float(s_v[col]); // fusion bias
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
template <typename T>
void launch_m256n64k32(half *a, half *b, half *c, half *v, const size_t M, const size_t N, const size_t K, cudaStream_t stream)
{
    const int BM = 256, BN = 64;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY);
    m256n64k32<<<gridDim, blockDim, 0, stream>>>(a, b, c, v, M, N, K);
}

/**************************************************************/
template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (size_t i = 0; i < num_warmups; i++)
    {
        bound_function(stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (size_t i = 0; i < num_repeats; i++)
    {
        bound_function(stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float const latency = time / num_repeats;

    return latency;
}

template <typename T>
bool is_equal(T const *data_1, T const *data_2, size_t size, const float atol)
{
    for (size_t i{0}; i < size; ++i)
    {
        float diff = abs(static_cast<float>(data_1[i]) - static_cast<float>(data_2[i]));
        if (diff > atol)
        {
            std::cout << "wrong at: " << i << "\n\t| "
                      << static_cast<float>(data_1[i])
                      << " <=> "
                      << static_cast<float>(data_2[i])
                      << " diff: " << diff << std::endl;
            return false;
        }
    }
    return true;
}
template <typename T>
bool verify_gemm_implementation(
    std::function<void(T *, T *, T *, T *, size_t, size_t, size_t, cudaStream_t)> launch_function,
    const size_t M, const size_t N, const size_t K, const float atol)
{
    // Fixed random seed for reproducibility
    std::mt19937 gen{0};
    cudaStream_t stream;
    std::vector<T> A(M * K, 0.5f);
    std::vector<T> B(K * N, 0.5f);
    std::vector<T> C(M * N, 0.0f);
    std::vector<T> gemm_reference(M * N, 0.0f);
    std::vector<T> v(N, 0.0f);
    std::uniform_real_distribution<float> uniform_dist(-1, 1);
    for (size_t i = 0; i < M * K; ++i)
        A[i] = static_cast<T>(uniform_dist(gen));
    for (size_t i = 0; i < N * K; ++i)
        B[i] = static_cast<T>(uniform_dist(gen));
    for (size_t i = 0; i < N; i++)
        v[i] = static_cast<T>(uniform_dist(gen));
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            float tmp = static_cast<float>(v[j]);
            for (size_t k = 0; k < K; ++k)
                tmp += (static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]));
            gemm_reference[i * N + j] = static_cast<T>(tmp);
        }
    }

    T *d_a, *d_b, *d_c, *d_v;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, N * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpy(d_a, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B.data(), N * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, C.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    launch_function(d_a, d_b, d_c, d_v, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK_LAST_CUDA_ERROR();
    CUDA_CHECK(cudaMemcpy(C.data(), d_c, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    bool const correctness = is_equal(C.data(), gemm_reference.data(), M * N, atol);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return correctness;
}

template <typename T>
float profile_gemm_implementation(std::function<void(T *, T *, T *, T *, size_t, size_t, size_t, cudaStream_t)> _function,
                                  const size_t M, const size_t N, const size_t K, const int num_repeats, const int num_warmups)
{
    cudaStream_t stream;
    T *d_a, *d_b, *d_c, *d_v;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, N * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const wrapped_function = std::bind(_function,
                                                                         d_a, d_b, d_c, d_v,
                                                                         M, N, K,
                                                                         std::placeholders::_1);
    float const function_latency = measure_performance(wrapped_function, stream, num_repeats, num_warmups);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return function_latency;
}

void print_latencty(std::string const &kernel_name, float latency)
{
    std::cout << kernel_name << ": " << std::fixed << std::setprecision(2)
              << latency << " ms" << std::endl;
}

void sgemm_performance_test(const size_t M, const size_t N, const size_t K, const int num_repeats, const int num_warmups)
{
    float const latency_gemm = profile_gemm_implementation<float>(&launch_m128n128k8<float>, M, N, K, num_repeats, num_warmups);
    std::stringstream ss;
    ss << "SGEMM [" << M << " " << N << " " << K << "]: " << "latency";
    print_latencty(ss.str(), latency_gemm);
}

void hgemm_performance_test(const size_t M, const size_t N, const size_t K, const int num_repeats, const int num_warmups)
{
    float const latency_gemm = profile_gemm_implementation<half>(&launch_m128n128k32<half>, M, N, K, num_repeats, num_warmups);
    std::stringstream ss;
    ss << "HGEMM [" << M << " " << N << " " << K << "] " << "latency: ";
    print_latencty(ss.str(), latency_gemm);
}

int main(int argc, char **argv)
{
    if (verify_gemm_implementation<float>(&launch_m128n128k8<float>, 1024, 1024, 1024, 1e-4))
        std::cout << "Pass" << std::endl;
    sgemm_performance_test(2048, 2048, 2048, 1, 1);
    sgemm_performance_test(4096, 4096, 4096, 1, 1);
    sgemm_performance_test(8192, 8192, 8192, 1, 1);
    sgemm_performance_test(16384, 16384, 16384, 1, 1);

    // if (verify_gemm_implementation<half>(&launch_m256n64k32<half>, 256, 256, 256, 2e-1))
    //     std::cout << "Pass" << std::endl;

    if (verify_gemm_implementation<half>(&launch_m128n128k32<half>, 1024, 1024, 1024, 2e-1))
        std::cout << "Pass" << std::endl;

    hgemm_performance_test(2048, 2048, 2048, 1, 1);
    hgemm_performance_test(4096, 4096, 4096, 1, 1);
    hgemm_performance_test(8192, 8192, 8192, 1, 1);
    hgemm_performance_test(16384, 16384, 16384, 1, 1);
    return 0;
}
