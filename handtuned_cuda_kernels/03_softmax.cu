/*
    Softmax: source from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh
    nvcc 03_softmax.cu -o softmax
    ./softmax
    rows = 32 * 64 * 512

    cols = 512
    [float]       Softmax: 2.349 ms TFlops: 0.914
    [half]        Softmax: 1.178 ms TFlops: 1.823
    [nv_bfloat16] Softmax: 1.336 ms TFlops: 1.607

    cols = 1024
    [float]       Softmax: 4.741 ms TFlops: 0.906
    [half]        Softmax: 3.068 ms TFlops: 1.400
    [nv_bfloat16] Softmax: 3.633 ms TFlops: 1.182
*/

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <time.h>
#include <math_constants.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
using namespace std;

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

constexpr int kWarpSize = 32;

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};
template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

// 每个 Warp 处理一行或两行元素，每行的Reduce操作 需要做 Warp 内的 Reduce 操作，
// 我们实现 WarpAllReduce 来完成 Warp 内各线程间的求 Global Max 和 Global Sum 操作，
// WarpAllReduce 是利用Warp级别原语 __shfl_xor_sync 实现的，代码如下.
template <template <typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val)
{
    // 是一个小于等于32的2的幂次
    // 假设thread_group_width=32
    // 第1轮: mask=32/2=16
    // 0^16=16, 1^16=17, 2^16=18, ..., 15^16=31
    // 第2轮: mask=16/2=8
    // 0^8=8, 1^8=9, 2^8=10, ..., 7^8=15, ***
    // 第3轮: mask=8/2=4
    // 0^4=4, 1^4=5, 2^4=6, 3^4=7, ***
    // 第4轮: mask=4/2=2
    // 0^2=2, 1^2=3, ***
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// BlockReduce 使用 cub 进行实现
template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val)
{
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0)
    {
        result_broadcast = result;
    }
    __syncthreads();
    return result_broadcast;
}

// 定义各种数据类型下面的 inf 值
template <typename T>
__inline__ __device__ T Inf();
template <>
__inline__ __device__ float Inf<float>()
{
    return CUDART_INF_F;
}
template <>
__inline__ __device__ double Inf<double>()
{
    return CUDART_INF;
}

// 定义 exp 函数，OF_SOFTMAX_USE_FAST_MATH 这个宏表示开启 FAST_MATH 的 exp
template <typename T>
__inline__ __device__ T Exp(T x);
template <>
__inline__ __device__ float Exp<float>(float x)
{
#ifdef OF_SOFTMAX_USE_FAST_MATH
    return __expf(x);
#else
    return exp(x);
#endif
}
template <>
__inline__ __device__ double Exp<double>(double x)
{
    return exp(x);
}

// 定义 div 函数，OF_SOFTMAX_USE_FAST_MATH 这个宏表示开启 FAST_MATH 的 div
template <typename T>
__inline__ __device__ T Div(T a, T b);
template <>
__inline__ __device__ float Div<float>(float a, float b)
{
#ifdef OF_SOFTMAX_USE_FAST_MATH
    return __fdividef(a, b);
#else
    return a / b;
#endif
}
template <>
__inline__ __device__ double Div<double>(double a, double b)
{
    return a / b;
}

// 定义 log 函数，OF_SOFTMAX_USE_FAST_MATH 这个宏表示开启 FAST_MATH 的 log
template <typename T>
__inline__ __device__ T Log(T x);
template <>
__inline__ __device__ float Log<float>(float x)
{
#ifdef OF_SOFTMAX_USE_FAST_MATH
    return __logf(x);
#else
    return log(x);
#endif
}
template <>
__inline__ __device__ double Log<double>(double x)
{
    return log(x);
}

// 对于 cuda kernel 来说, 启动多少个block (grid_size) 来做计算
// [如何设置CUDA Kernel中的grid_size和block_size](https://mp.weixin.qq.com/s/1_ao9xM6Qk3JaavptChXew)
inline cudaError_t GetNumBlocks(int64_t block_size,
                                int64_t max_blocks,
                                int64_t waves,
                                int *num_blocks)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    *num_blocks =
        std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}

// 定义默认的计算类型，一般的数据类型进行 softmax 计算时计算类型就是自己，
// 而对于 half（或者bfp16） 来说我们往往需要将计算类型 fallback 到 float 来避免精度损失
template <typename T>
struct DefaultComputeType
{
    using type = T;
};

template <>
struct DefaultComputeType<half>
{
    using type = float;
};

#if CUDA_VERSION >= 11000
template <>
struct DefaultComputeType<nv_bfloat16>
{
    using type = float;
};
#endif // CUDA_VERSION >= 11000

// GetPackType 结构体中使用了 std::aligned_storage 先声明了一个内存对齐的数据类型 type,
// 注意这个 type 的内存长度为 pack_size * sizeof(T). 然后这里的 T 是我们需要进行 Pack
// 的数据类型，而 pack_size 则表示我们需要 Pack 的元素个数. 接下来我们看到 Pack 联合体中
// 声明了 storage 和 elem 两个数组, 它们共用同一段对齐的内存. 然后 Pack 联合体的入口有一个
// 检查: static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
// 这是用来判断我们之前声明的 type 的内存长度是否符合预期.
template <typename T, int N>
struct GetPackType
{
    using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int N>
union Pack
{
    static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
    __device__ Pack()
    {
        // do nothing
    }
    PackType<T, N> storage;
    T elem[N];
};

// 下面分别定义了两个代表输入输出的数据结构
template <typename SRC, typename DST>
struct DirectLoad
{
    DirectLoad(const SRC *src, int64_t row_size) : src(src), row_size(row_size) {}
    template <int pack_size>
    __device__ void load(DST *dst, int64_t row, int64_t col) const // const成员函数中不能修改成员变量
    {
        Pack<SRC, pack_size> pack;
        const int64_t offset = (row * row_size + col) / pack_size;
        pack.storage = *(reinterpret_cast<const PackType<SRC, pack_size> *>(src) + offset);
#pragma unroll
        for (int i = 0; i < pack_size; ++i)
        {
            dst[i] = static_cast<DST>(pack.elem[i]);
        }
    }
    const SRC *src;
    int64_t row_size;
};

template <typename SRC, typename DST>
struct DirectStore
{
    DirectStore(DST *dst, int64_t row_size) : dst(dst), row_size(row_size) {}
    template <int N>
    __device__ void store(const SRC *src, int64_t row, int64_t col)
    {
        Pack<DST, N> pack;
        const int64_t offset = (row * row_size + col) / N;
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            pack.elem[i] = static_cast<DST>(src[i]);
        }
        *(reinterpret_cast<PackType<DST, N> *>(dst) + offset) = pack.storage;
    }
    DST *dst;
    int64_t row_size;
};

// 枚举类, kSoftmax和kLogSoftmax可选
enum class Algorithm
{
    kSoftmax = 0,
    kLogSoftmax = 1,
};

// 使用 `load.template load<pack_size>(ptr, row_id, col_id);`
// 和`store.template store<pack_size>(ptr, row_id, col_id);` 进行读取和写入
// 使用LOAD和STORE有两个好处:
// 1. 可以在CUDA Kernel中只关心计算类型ComputeType，而不用关心具体的数据类型T
// 2. 只需要加几行代码就可以快速支持Softmax和其他Kernel Fuse，减少带宽需求，提升整体性能
// 普通的SoftmaxKernel直接使用DirectLoad和DirectStore
// FusedSoftmaxKernel如FusedScaleSoftmaxDropoutKernel
// 只需要定义一个ScaleLoad结构和一个DropoutStore结构用于对输入x做Scale预处理和对输出y做Dropout后处理
// ComputeType代表计算类型:
// 1. pack_size代表向量化访存操作的pack元素的个数, 我们将几个元素pack起来读写, 提升带宽利用率
// 2. cols_per_thread代表每个线程处理的元素个数
// 3. thread_group_width代表处理元素的线程组的宽度
// 当cols > pack_size * warp_size时, thread_group_width就是warp_size, 即32.
// 当cols < pack_size * warp_size时, 就根据cols大小用1/2个warp或1/4个warp来处理每行的元素.
// 采用更小的thread_group_width后, WarpAllReduce需要执行的轮次也相应减少.
// rows_per_access 代表每个线程组一次处理的行数, 当cols较小, thread_group_width 不是 warp_size=32时,
// 若rows能被2整除, 我们就让每个线程处理2行来增加指令并行度, 从而提升性能.
// padding 代表当前是否做了padding, 若 cols 不是 warp_size 的整数倍, 我们会把它padding到最近的整数倍处理.

// kernel 1: cols<=1024
// 一行的数据全分散给多个线程的寄存器中 cols_per_thread * thread_group_width >= cols
// 每个线程分配 rows_per_access * cols_per_thread 大小的寄存器存原始输入
// 每个线程分配 rows_per_access 个寄存器, 存对应行的最大值
// 将 packed 输入读到寄存器中, 每个线程在取数的过程中, 求出线程各自计算的数据的最大值
// maxReduce 得到一行的最大值 x_m
// 计算 e^(x_i-x_m) 和 sum e^(x_i-x_m)
// sumReduce 得到一行的 sum e^(x_i-x_m)
// 每个线程计算 e^(x_i-x_m)/sum
// 总共启动 blockDim.x 个 block 来一次性处理多行, 每个线程 += gridDim.x 处理多行
template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
          int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols)
{
    // 我们需要保证每个线程处理的元素个数可以被 pack_size 整除
    static_assert(cols_per_thread % pack_size == 0, "");
    // 处理元素的线程组的宽度需要小于等于 kWarpSize, 并且需要被 kWarpSize 整除
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");
    // 每个线程处理的 pack 后的元素个数
    constexpr int num_packs = cols_per_thread / pack_size;
    // 需要保证 cols <= 每个线程处理的元素个数 * 处理元素的线程组的宽度
    // 因为这个地方是每个 Warp 处理一行或两行元素
    assert(cols <= cols_per_thread * thread_group_width);
    // 开一块 rows_per_access * cols_per_thread 的寄存器
    ComputeType buf[rows_per_access][cols_per_thread];

    // dim3 block_dim(thread_group_width, thread_groups_per_block);
    // grid 是一维, block 是二维的
    // thread_group_width 表示block中每个线程组的大小 可以理解为设置的 warp_size
    // thread_groups_per_block 表示每个 block 里面有多少个线程组
    // 如果 thread_group_width = 32, thread_groups_per_block = 4
    // 那么每个 block 的总线程数是 128

    // 获取全局线程组的数量
    const int num_global_thread_group = gridDim.x * blockDim.y;
    // 获取全局的线程组id
    // blockIdx.x 表示在第几个block, blockDim.y 表示每个 block 里面有多少个线程组
    // threadIdx.y 表示在当前 block 里面的第几个线程组
    const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;

    // lane id 表示当前线程在当前block的所在线程组的 id
    const int lane_id = threadIdx.x;
    // step 表示循环计数器大小为 全局线程组的数量 * 每个线程组一次处理的行数
    const int64_t step = num_global_thread_group * rows_per_access;
    // for 循环的开始为 row = 全局的线程组id * 每个线程组一次处理的行数, 结束为总行数
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step)
    {
        // 开辟一块寄存器记录当前线程组处理的每一行的最大值
        ComputeType thread_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            // 把 thread_max[row_id] 初始化为 -inf
            thread_max[row_id] = -Inf<ComputeType>();
            // 获取第 row_id 行的共享内存数据
            ComputeType *row_buf = buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < num_packs; ++pack_id)
            {
                // pack的偏移量
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols)
                {
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
                    for (int i = 0; i < pack_size; ++i)
                    {
                        thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                    }
                }
                else // padding位置的元素置为-inf
                {
#pragma unroll
                    for (int i = 0; i < pack_size; ++i)
                    {
                        row_buf[pack_offset + i] = -Inf<ComputeType>();
                    }
                }
            }
        }
        // 开辟一块寄存器记录属于同一个warp的线程组的每一行的最大值
        // 也就是需要进行一次warpReduce max
        ComputeType warp_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
        }
        // 开辟一块共享内存记录当前线程组处理的每一行的sum
        ComputeType thread_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            thread_sum[row_id] = 0;
            ComputeType *row_buf = buf[row_id];
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
                    thread_sum[row_id] += row_buf[i];
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    row_buf[i] -= warp_max[row_id];
                    thread_sum[row_id] += Exp(row_buf[i]);
                }
                else
                {
                    __trap(); // 内核的执行被中止并在主机程序中引发中断.
                }
            }
        }
        // 开辟一块共享内存记录属于同一个warp的线程组的每一行的sum (注意这里考虑了指数运算的安全性，
        // 实际上求的是sum{exp(x_i-max{x_i})})，也就是需要进行一次warpReduce sum
        ComputeType warp_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
        }
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            ComputeType *row_buf = buf[row_id];
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    row_buf[i] -= Log(warp_sum[row_id]);
                }
                else
                {
                    __trap(); // 内核的执行被中止并在主机程序中引发中断.
                }
            }
#pragma unroll
            for (int i = 0; i < num_packs; ++i)
            {
                const int col = (i * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols)
                {
                    store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
                }
            }
        }
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
          int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                         const int64_t rows, const int64_t cols)
{
    // 每个 Block 里面设置 128 个线程
    constexpr int block_size = 128;
    // GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数个 block，
    // 因为每个 block 的计算量相等，所以所有 SM 应几乎同时完成这些 block 的计算，
    // 然后处理下一批，这其中的每一批被称之为一个 wave.
    // grid_size 设置为可以满足足够多的 wave，也就是这里定义的 waves
    constexpr int waves = 32;
    // block_size 需要整除处理元素的线程组的宽度
    static_assert(block_size % thread_group_width == 0, "");
    // 每个 block 的线程组的个数，如果这里 thread_group_width = 32，那么 thread_groups_per_block = 4
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    // 根据数据的大小计算最多设置多少个 block ，就是数据的行数 / 每个线程组一次处理的行数
    const int64_t num_blocks =
        (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    // 根据上述的设置以及硬件本身的限制计算最终启动的 block 数
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                    rows_per_access, padding, algorithm>
        <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
          int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols)
{
    // 如果每个线程处理的元素个数 * 处理元素的线程组的宽度(warp_size)和cols相等，就不需要padding
    if (cols == cols_per_thread * thread_group_width)
    {
        return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                     thread_group_width, rows_per_access, false, algorithm>(
            stream, load, store, rows, cols);
    }
    else
    {
        return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                     thread_group_width, rows_per_access, true, algorithm>(
            stream, load, store, rows, cols);
    }
}

// pack_size == 1根据rows的行的奇偶性, 设置rows_per_access
template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols)
{
    if (cols <= 0)
    {
        return cudaErrorInvalidValue;
    }
// 这里是对比数据的列数(cols)和线程组的宽度*pack_size
// 如果满足条件就dispatch到对应的线程组宽度(warp_size)
// 注意这里是一个线程处理 pack_size 个元素
#define DEFINE_ONE_ELIF(thread_group_width)                                                              \
    else if (cols <= (thread_group_width) * pack_size)                                                   \
    {                                                                                                    \
        if (rows % 2 == 0)                                                                               \
        {                                                                                                \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                                  thread_group_width, 2, algorithm>(stream, load, store, \
                                                                                    rows, cols);         \
        }                                                                                                \
        else                                                                                             \
        {                                                                                                \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                                  thread_group_width, 1, algorithm>(stream, load, store, \
                                                                                    rows, cols);         \
        }                                                                                                \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
// 如果cols比较大, 上面的条件都不满足
// 那么直接一个线程处理col个元素, 而不是pack_size个元素
#define DEFINE_ONE_ELIF(col)                                                                          \
    else if (cols <= (col) * kWarpSize)                                                               \
    {                                                                                                 \
        return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                              algorithm>(stream, load, store, rows, cols);            \
    }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(7)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(9)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(11)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(13)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(15)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(17)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(19)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(21)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(23)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(25)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(27)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(29)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(31)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else
    {
        return cudaErrorInvalidValue;
    }
}

// pack_size == 2根据数据总的行的奇偶性, 设置rows_per_access
template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols)
{
    if (cols <= 0)
    {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                              \
    else if (cols <= (thread_group_width) * pack_size)                                                   \
    {                                                                                                    \
        if (rows % 2 == 0)                                                                               \
        {                                                                                                \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                                  thread_group_width, 2, algorithm>(stream, load, store, \
                                                                                    rows, cols);         \
        }                                                                                                \
        else                                                                                             \
        {                                                                                                \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                                  thread_group_width, 1, algorithm>(stream, load, store, \
                                                                                    rows, cols);         \
        }                                                                                                \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                          \
    else if (cols <= (col) * kWarpSize)                                                               \
    {                                                                                                 \
        return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                              algorithm>(stream, load, store, rows, cols);            \
    }
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else
    {
        return cudaErrorInvalidValue;
    }
}

// 根据cols的奇偶性, 设置pack_size
template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxWarpImplPackSize
{
    cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                           const int64_t cols)
    {
        if (cols % 2 == 0)
        {
            return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(stream, load,
                                                                                       store, rows, cols);
        }
        else
        {
            return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(stream, load,
                                                                                       store, rows, cols);
        }
    }
};

// kernel 1 接口
template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols)
{
    return DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, algorithm>()(stream, load, store,
                                                                                  rows, cols);
}

// kernel 2: 1024 < cols <= 4096
// 一个 Block 处理一行元素 将一行全放入shared memory
// 每个线程计算 pack_size 个元素的最大值 (+= block_size 的计算该行的多列) , 放在本地寄存器
// 利用 BlockAllReduce 完成对每个线程得到的最大值进行规约 (这里内部应该时开了shared memory然后在shared memory上做规约)
// 然后每个线程算出 pack_size 个元素的e^(x_i-x_m) (+= block_size 的计算该行的多列)
// 再利用 BlockAllReduce 得到该行的sum e^(x_i-x_m)
// 每个线程再算出 e^(x_i-x_m)/sum
template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
          Algorithm algorithm>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols)
{
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto *buf = reinterpret_cast<ComputeType *>(shared_buf);
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    // 一个 Block 处理一行元素
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        // 当前线程的最大值初始化为 -inf
        ComputeType thread_max = -Inf<ComputeType>();
        // 以向量化的方式加载一行数据，然后执行pack reduce操作
        // 这里的 pack reduce操作, 在 https://zhuanlan.zhihu.com/p/596012674 最后一节也有介绍
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                buf[i * num_packs + pack_id] = pack[i];
                thread_max = max(thread_max, pack[i]);
            }
        }
        // 执行block reduce获取当前行（由一个 Block 进行处理）的最大值
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
        ComputeType thread_sum = 0;
        for (int col = tid; col < cols; col += block_size)
        {
            if (algorithm == Algorithm::kSoftmax)
            {
                const ComputeType exp_x = Exp(buf[col] - row_max);
                buf[col] = exp_x;
                thread_sum += exp_x;
            }
            else
            {
                const ComputeType x = buf[col] - row_max;
                buf[col] = x;
                thread_sum += Exp(x);
            }
        }
        // 同理，获得当前行的sum
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        // 计算结果并写回到全局内存中
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType pack[pack_size];
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
                }
                else
                {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
          Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem,
                                              const int64_t rows, const int64_t cols)
{
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

// 执行的主体循环逻辑如下, 根据 num_cols 算出需要的 Shared Memory 大小作为 Launch Kernel 参数,
// 借助 Shared Memory 保存输入, 后续计算直接从 Shared Memory 读取.
// 由于 SM 内的 Shared Memory 资源同样有限, 因此当 num_cols超过一定范围, kernel 启动时申请 Shared Memory 超过最大限制,
// 就会出现无法启动的问题, 因此, 仅在调用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 返回值大于0时采用 Shared Memory 方案.
// 此外, 需要注意的是, 由于 Block 内线程要做同步, 当 SM 中正在调度执行的一个 Block 到达同步点时, SM 内可执行 Warp 逐渐减少,
// 若同时执行的 Block 只有一个, 则 SM 中可同时执行的 Warp 会在此时逐渐降成0, 会导致计算资源空闲, 造成浪费
// 若此时同时有其他 Block 在执行, 则在一个 Block 到达同步点时仍然有其他 Block 可以执行
// 当 block_size 越小时, SM 可同时调度的 Block 越多, 因此在这种情况下 block_size 越小越好.
// 但是当在调大 block_size, SM 能同时调度的 Block 数不变的情况下, block_size 应该是越大越好, 越大就有越好的并行度.
// 因此代码中在选择 block_size 时, 对不同 block_size 都计算了 cudaOccupancyMaxActiveBlocksPerMultiprocessor, 若结果相同, 使用较大的 block_size.
template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                            STORE store, const int64_t rows,
                                                            const int64_t cols, bool *success)
{
    // 设置4个不同的block_size
    constexpr int block_size_conf_1 = 128;
    constexpr int block_size_conf_2 = 256;
    constexpr int block_size_conf_3 = 512;
    constexpr int block_size_conf_4 = 1024;
    // 计算第二种方案需要的共享内存大小
    const size_t smem = cols * sizeof(ComputeType);
    int max_active_blocks_conf_1;
    {
        // 占用计算器API cudaOccupancyMaxActiveBlocksPerMultiprocessor
        // 可以根据 kernel 的 block 大小和共享内存使用情况提供占用率预测
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_1,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
            block_size_conf_1, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_1 <= 0)
    {
        *success = false;
        return cudaSuccess;
    }
    int max_active_blocks_conf_4;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_4,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
            block_size_conf_4, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_4 == max_active_blocks_conf_1)
    {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4,
                                          algorithm>(stream, load, store, smem, rows, cols);
    }
    int max_active_blocks_conf_3;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_3,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
            block_size_conf_3, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_3 == max_active_blocks_conf_1)
    {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3,
                                          algorithm>(stream, load, store, smem, rows, cols);
    }
    int max_active_blocks_conf_2;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_2,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
            block_size_conf_2, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_2 == max_active_blocks_conf_1)
    {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2,
                                          algorithm>(stream, load, store, smem, rows, cols);
    }
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1,
                                      algorithm>(stream, load, store, smem, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxBlockSMemImplPackSize
{
    cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                           const int64_t cols, bool *success)
    {
        if (cols % 2 == 0)
        {
            return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(
                stream, load, store, rows, cols, success);
        }
        else
        {
            return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(
                stream, load, store, rows, cols, success);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                   const int64_t rows, const int64_t cols,
                                                   bool *success)
{
    return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
        stream, load, store, rows, cols, success);
}

// kernel 3: cols>=4096
// TODO: 不知道这种情况下用online-softmax会不会好点
// 一个 Block 处理一行元素
// 不再用 Shared Memory 缓存输入, 在每次计算时重新读输入
// 这种实现没有最大 num_cols的限制，可以支持任意大小.
// 需要注意的是，在这种实现中，block_size 应该设越大越好，
// block_size 越大，SM 中能同时并行执行的 Block 数就越少，
// 对 cache 的需求就越少，就有更多机会命中 Cache，
// 多次读x不会多次访问 Global Memory，因此在实际测试中，
// 在能利用 Cache 情况下，有效带宽不会因为读3次x而降低几倍.
template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
          Algorithm algorithm>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                         const int64_t cols)
{
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        ComputeType thread_max = -Inf<ComputeType>();
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                thread_max = max(thread_max, pack[i]);
            }
        }
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                thread_sum += Exp(pack[i] - row_max);
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    pack[i] = Div(Exp(pack[i] - row_max), row_sum);
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    pack[i] = (pack[i] - row_max) - Log(row_sum);
                }
                else
                {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols)
{
    // 每个 Block 使用 1024 个线程
    constexpr int block_size = 1024;
    // waves 需要满足32组
    constexpr int waves = 32;
    // 根据 BlockSize 以及硬件参数计算 Block 数量
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    // 启动第3个实现的 cuda kernel
    SoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxBlockUncachedImplPackSize
{
    cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                           const int64_t cols)
    {
        // 如果 cols % 2 == 0，就执行 pack2，否则不 pack
        if (cols % 2 == 0)
        {
            return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 2, algorithm>(
                stream, load, store, rows, cols);
        }
        else
        {
            return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 1, algorithm>(
                stream, load, store, rows, cols);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols)
{
    return DispatchSoftmaxBlockUncachedImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
        stream, load, store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                const int64_t cols)
{
    if (cols < 1024)
    {
        return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
            stream, load, store, rows, cols);
    }
    else
    {
        bool dispatch_smem_impl_success;
        {
            // 使用实现2的shared memory版本
            cudaError_t err =
                TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
                    stream, load, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (!dispatch_smem_impl_success)
        {
            return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
                stream, load, store, rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                const int64_t cols)
{
    return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
        stream, load, store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                   const int64_t cols)
{
    if (cols <= 1024)
    {
        return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
            stream, load, store, rows, cols);
    }
    else
    {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err =
                TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
                    stream, load, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (!dispatch_smem_impl_success)
        {
            return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
                stream, load, store, rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                   const int64_t cols)
{
    return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
        stream, load, store, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
          Algorithm algorithm>
__global__ void SoftmaxGradWarpImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
                                    const int64_t cols)
{
    static_assert(cols_per_thread % pack_size == 0, "");
    constexpr int pack_per_thread = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * thread_group_width);
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");
    ComputeType y_buf[rows_per_access][cols_per_thread];
    ComputeType dy_buf[rows_per_access][cols_per_thread];
    const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_global_thread_group = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step)
    {
        ComputeType thread_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            thread_sum[row_id] = 0;
            ComputeType *row_y_buf = y_buf[row_id];
            ComputeType *row_dy_buf = dy_buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id)
            {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols)
                {
                    load_y.template load<pack_size>(row_y_buf + pack_offset, row + row_id, col);
                    load_dy.template load<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
#pragma unroll
                    for (int i = 0; i < pack_size; ++i)
                    {
                        if (algorithm == Algorithm::kSoftmax)
                        {
                            thread_sum[row_id] += row_y_buf[pack_offset + i] * row_dy_buf[pack_offset + i];
                        }
                        else if (algorithm == Algorithm::kLogSoftmax)
                        {
                            thread_sum[row_id] += row_dy_buf[pack_offset + i];
                        }
                        else
                        {
                            __trap();
                        }
                    }
                }
            }
        }
        ComputeType warp_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
        }
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
        {
            ComputeType *row_y_buf = y_buf[row_id];
            ComputeType *row_dy_buf = dy_buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id)
            {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols)
                {
                    for (int i = 0; i < pack_size; ++i)
                    {
                        if (algorithm == Algorithm::kSoftmax)
                        {
                            row_dy_buf[pack_offset + i] =
                                (row_dy_buf[pack_offset + i] - warp_sum[row_id]) * row_y_buf[pack_offset + i];
                        }
                        else if (algorithm == Algorithm::kLogSoftmax)
                        {
                            row_dy_buf[pack_offset + i] -= Exp(row_y_buf[pack_offset + i]) * warp_sum[row_id];
                        }
                        else
                        {
                            __trap();
                        }
                    }
                    store.template store<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
                }
            }
        }
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
          Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                             STORE store, const int64_t rows, const int64_t cols)
{
    constexpr int block_size = 128;
    constexpr int waves = 32;
    static_assert(block_size % thread_group_width == 0, "");
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    const int64_t num_blocks =
        (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    SoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread,
                        thread_group_width, rows_per_access, padding, algorithm>
        <<<grid_dim_x, block_dim, 0, stream>>>(load_y, load_dy, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          int cols_per_thread, int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImplPadding(cudaStream_t stream, LOAD_Y load_y,
                                                      LOAD_DY load_dy, STORE store,
                                                      const int64_t rows, const int64_t cols)
{
    if (cols == cols_per_thread * thread_group_width)
    {
        return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                         cols_per_thread, thread_group_width, rows_per_access, false,
                                         algorithm>(stream, load_y, load_dy, store, rows, cols);
    }
    else
    {
        return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                         cols_per_thread, thread_group_width, rows_per_access, true,
                                         algorithm>(stream, load_y, load_dy, store, rows, cols);
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols)
{
    if (cols <= 0)
    {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                           \
    else if (cols <= (thread_group_width) * pack_size)                                                \
    {                                                                                                 \
        if (rows % 2 == 0)                                                                            \
        {                                                                                             \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                      pack_size, thread_group_width, 2, algorithm>(   \
                stream, load_y, load_dy, store, rows, cols);                                          \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                      pack_size, thread_group_width, 1, algorithm>(   \
                stream, load_y, load_dy, store, rows, cols);                                          \
        }                                                                                             \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                           \
    else if (cols <= (col) * kWarpSize)                                                                \
    {                                                                                                  \
        return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, \
                                                  kWarpSize, 1, algorithm>(stream, load_y, load_dy,    \
                                                                           store, rows, cols);         \
    }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(7)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(9)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(11)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(13)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(15)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(17)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(19)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(21)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(23)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(25)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(27)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(29)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(31)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else
    {
        return cudaErrorInvalidValue;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols)
{
    if (cols <= 0)
    {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                           \
    else if (cols <= (thread_group_width) * pack_size)                                                \
    {                                                                                                 \
        if (rows % 2 == 0)                                                                            \
        {                                                                                             \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                      pack_size, thread_group_width, 2, algorithm>(   \
                stream, load_y, load_dy, store, rows, cols);                                          \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                      pack_size, thread_group_width, 1, algorithm>(   \
                stream, load_y, load_dy, store, rows, cols);                                          \
        }                                                                                             \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                           \
    else if (cols <= (col) * kWarpSize)                                                                \
    {                                                                                                  \
        return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, \
                                                  kWarpSize, 1, algorithm>(stream, load_y, load_dy,    \
                                                                           store, rows, cols);         \
    }
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else
    {
        return cudaErrorInvalidValue;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
          Algorithm algorithm>
struct DispatchSoftmaxGradWarpImplPackSize
{
    cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                           const int64_t rows, const int64_t cols)
    {
        if (cols % 2 == 0)
        {
            return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(
                stream, load_y, load_dy, store, rows, cols);
        }
        else
        {
            return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(
                stream, load_y, load_dy, store, rows, cols);
        }
    }
};

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
          Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                               STORE store, const int64_t rows,
                                               const int64_t cols)
{
    return DispatchSoftmaxGradWarpImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, algorithm>()(
        stream, load_y, load_dy, store, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockSMemImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                         const int64_t rows, const int64_t cols)
{
    extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
    auto *y_buf = reinterpret_cast<ComputeType *>(grad_shared_buf);
    auto *dy_buf = y_buf + cols;
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType y_pack[pack_size];
            ComputeType dy_pack[pack_size];
            load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
            load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                y_buf[i * num_packs + pack_id] = y_pack[i];
                dy_buf[i * num_packs + pack_id] = dy_pack[i];
                if (algorithm == Algorithm::kSoftmax)
                {
                    thread_sum += y_pack[i] * dy_pack[i];
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    thread_sum += dy_pack[i];
                }
                else
                {
                    __trap();
                }
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType pack[pack_size];
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    pack[i] = (dy_buf[i * num_packs + pack_id] - row_sum) * y_buf[i * num_packs + pack_id];
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    pack[i] = dy_buf[i * num_packs + pack_id] - Exp(y_buf[i * num_packs + pack_id]) * row_sum;
                }
                else
                {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          int block_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y,
                                                  LOAD_DY load_dy, STORE store, int smem,
                                                  const int64_t rows, const int64_t cols)
{
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, smem, stream>>>(load_y, load_dy, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImplBlockSize(cudaStream_t stream, LOAD_Y load_y,
                                                                LOAD_DY load_dy, STORE store,
                                                                const int64_t rows,
                                                                const int64_t cols, bool *success)
{
    constexpr int block_size_conf_1 = 128;
    constexpr int block_size_conf_2 = 256;
    constexpr int block_size_conf_3 = 512;
    constexpr int block_size_conf_4 = 1024;
    const size_t smem = cols * sizeof(ComputeType) * 2;
    int max_active_blocks_conf_1;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_1,
            SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1,
                                     algorithm>,
            block_size_conf_1, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_1 <= 0)
    {
        *success = false;
        return cudaSuccess;
    }
    int max_active_blocks_conf_4;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_4,
            SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_4,
                                     algorithm>,
            block_size_conf_4, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_4 == max_active_blocks_conf_1)
    {
        *success = true;
        return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                              block_size_conf_4, algorithm>(stream, load_y, load_dy,
                                                                            store, smem, rows, cols);
    }
    int max_active_blocks_conf_3;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_3,
            SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_3,
                                     algorithm>,
            block_size_conf_3, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_3 == max_active_blocks_conf_1)
    {
        *success = true;
        return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                              block_size_conf_3, algorithm>(stream, load_y, load_dy,
                                                                            store, smem, rows, cols);
    }
    int max_active_blocks_conf_2;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_2,
            SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_2,
                                     algorithm>,
            block_size_conf_2, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if (max_active_blocks_conf_2 == max_active_blocks_conf_1)
    {
        *success = true;
        return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                              block_size_conf_2, algorithm>(stream, load_y, load_dy,
                                                                            store, smem, rows, cols);
    }
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_1, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
          Algorithm algorithm>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize
{
    cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                           const int64_t rows, const int64_t cols, bool *success)
    {
        if (cols % 2 == 0)
        {
            return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 2,
                                                                algorithm>(stream, load_y, load_dy, store,
                                                                           rows, cols, success);
        }
        else
        {
            return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 1,
                                                                algorithm>(stream, load_y, load_dy, store,
                                                                           rows, cols, success);
        }
    }
};

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
          Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y,
                                                       LOAD_DY load_dy, STORE store,
                                                       const int64_t rows, const int64_t cols,
                                                       bool *success)
{
    return TryDispatchSoftmaxGradBlockSMemImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                       algorithm>()(stream, load_y, load_dy, store,
                                                                    rows, cols, success);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockUncachedImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                             const int64_t rows, const int64_t cols)
{
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType y_pack[pack_size];
            ComputeType dy_pack[pack_size];
            load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
            load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);

#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    thread_sum += y_pack[i] * dy_pack[i];
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    thread_sum += dy_pack[i];
                }
                else
                {
                    __trap();
                }
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
        {
            ComputeType y_pack[pack_size];
            ComputeType dy_pack[pack_size];
            load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
            load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i)
            {
                if (algorithm == Algorithm::kSoftmax)
                {
                    dy_pack[i] = (dy_pack[i] - row_sum) * y_pack[i];
                }
                else if (algorithm == Algorithm::kLogSoftmax)
                {
                    dy_pack[i] -= Exp(y_pack[i]) * row_sum;
                }
                else
                {
                    __trap();
                }
            }
            store.template store<pack_size>(dy_pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                      LOAD_DY load_dy, STORE store,
                                                      const int64_t rows, const int64_t cols)
{
    constexpr int block_size = 1024;
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    SoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size,
                                 algorithm>
        <<<grid_dim_x, block_size, 0, stream>>>(load_y, load_dy, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
          Algorithm algorithm>
struct DispatchSoftmaxGradBlockUncachedImplPackSize
{
    cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                           const int64_t rows, const int64_t cols)
    {
        if (cols % 2 == 0 && cols > kWarpSize)
        {
            return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(
                stream, load_y, load_dy, store, rows, cols);
        }
        else
        {
            return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(
                stream, load_y, load_dy, store, rows, cols);
        }
    }
};

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
          Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                        LOAD_DY load_dy, STORE store,
                                                        const int64_t rows, const int64_t cols)
{
    return DispatchSoftmaxGradBlockUncachedImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                        algorithm>()(stream, load_y, load_dy, store,
                                                                     rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                    const int64_t rows, const int64_t cols)
{
    if (cols <= 1024)
    {
        return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(
            stream, load_y, load_dy, store, rows, cols);
    }
    else
    {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                                  Algorithm::kSoftmax>(
                stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (!dispatch_smem_impl_success)
        {
            return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                        Algorithm::kSoftmax>(stream, load_y, load_dy,
                                                                             store, rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                    const int64_t rows, const int64_t cols)
{
    return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                Algorithm::kSoftmax>(stream, load_y, load_dy, store,
                                                                     rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                       const int64_t rows, const int64_t cols)
{
    if (cols <= 1024)
    {
        return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(
            stream, load_y, load_dy, store, rows, cols);
    }
    else
    {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                                  Algorithm::kLogSoftmax>(
                stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (!dispatch_smem_impl_success)
        {
            return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                        Algorithm::kLogSoftmax>(stream, load_y, load_dy,
                                                                                store, rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                       const int64_t rows, const int64_t cols)
{
    return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                Algorithm::kLogSoftmax>(stream, load_y, load_dy,
                                                                        store, rows, cols);
}

template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream,
                          size_t num_repeats = 10,
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
void DispatchSoftmaxWrapper(T *output, T const *input, const int64_t rows, const int64_t cols, cudaStream_t stream)
{
    using ComputeType = typename DefaultComputeType<float>::type;
    DirectLoad<T, ComputeType> load(input, cols);
    DirectStore<ComputeType, T> store(output, cols);
    DispatchSoftmax<decltype(load), decltype(store), ComputeType>(stream, load, store, rows, cols);
}

template <typename T>
float softmax_benchmark(std::function<void(T *, T const *, const int64_t, const int64_t, cudaStream_t)> _function,
                        const int64_t rows, const int64_t cols, const bool verify)
{
    using ComputeType = typename DefaultComputeType<T>::type;
    T *input_host = (T *)malloc(rows * cols * sizeof(T));
    T *d_input, *d_output;
    const int64_t N = rows * cols;
    constexpr int num_repeats = 10;
    constexpr int num_warmups = 10;
    cudaStream_t stream;

    for (int64_t i = 0; i < N; i++)
        input_host[i] = 0.789;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(T)));
    cudaMemcpy(d_input, input_host, N * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const wrapped_function = std::bind(_function,
                                                                         d_output,
                                                                         d_input,
                                                                         rows, cols,
                                                                         std::placeholders::_1);
    float const function_latency = measure_performance(wrapped_function, stream, num_repeats, num_warmups);
    if (verify)
    {
        T *output_host = (T *)malloc(N * sizeof(T));
        cudaMemcpy(output_host, d_output, N * sizeof(T), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < N; i++)
            assert(fabs(static_cast<float>(output_host[i]) - (1 / (float)(cols))) < 1e-3 && "wrong");
        free(output_host);
        cout << "Check pass" << endl;
    }
    free(input_host);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return function_latency;
}

void print_latency(string const &kernel_name, float latency, const int64_t flops)
{
    cout << kernel_name << ": " << fixed << setprecision(3)
         << latency << " ms" << " TFlops: " << (float)flops / (latency * 0.001) / (1e12) << endl;
}

template <typename T>
void performance_test(const int64_t rows, const int64_t cols, const bool verify)
{
    const float latency_with_shm_bank_conflict = softmax_benchmark<T>(&DispatchSoftmaxWrapper<T>,
                                                                      rows, cols, verify);
    print_latency("Softmax", latency_with_shm_bank_conflict, 4 * rows * cols);
}

int main()
{
    const int64_t rows = 32 * 64 * 512;
    const int64_t cols = 1024;
    performance_test<float>(rows, cols, false);
    performance_test<half>(rows, cols, false);
    performance_test<nv_bfloat16>(rows, cols, false);
    return 0;
}