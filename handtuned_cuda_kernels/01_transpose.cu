/*
    24.12 yqy.
    为什么是矩阵转置?
        1. 加速核函数最简单的方法通常是使用shared memory, 但某些特定结构大小的数据和访问模式会造成造成bank conflict https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900
        2. 避免bank conflict的方法有两种padding和swizzle. padding是一种非常简单的技术, 通过在一"行"的尾部添加一些多余的数据, 使得原本在相同bank的数据因偏移而不在, 同一bank中, 但如果padding不当, 可能会引入miss alignment
        3. swizzle是一种更复杂的技术, 但不会造成, 并且允许你进行向量化访存
    什么情况需要swizzle/bank conflict发生的条件:
        1. shared memory形如T array[-][N], 且`N*sizeof(T)`是32的倍数
        2. 存在[tx][-]的访问模式或导致访问stride是128Byte的模式
        3. 一个warp内的线程发生上述情况(LDS.32)
            1. 如果warp中每个thread访问4Byte  [LDS.32],  请求不会被拆分,  需要在每1/1个warp内判断是否存在conflict
            2. 如果warp中每个thread访问8Byte  [LDS.64],  请求会被拆成2次, 需要在每1/2个warp内判断是否存在conflict
            3. 如果warp中每个thread访问16Byte [LDS.128], 请求会被拆成4次, 需要在每1/4个warp内判断是否存在conflict
    如何进行swizzle: [y][x] -> [y][x_swz]
        1. swizzle_size = N * sizeof(T)
        2. access granularity is sizeof(TC) byte
        3. i_chunk = (y*N + x)*sizeof(T) / sizeof(TC)
        4. x_chunk = i_chunk % (swizzle_size / sizeof(TC)) , y_chunk = i_chunk / (swizzle_size / sizeof(TC))
        5. x_chunk_swz = y_chunk ^ x_chunk
        6. x_swz = x_chunk_swz * sizeof(TC) / sizeof(T) % N + x % (sizeof(TC) / sizeof(T))

    Why is matrix transposition?
        1. The simplest way to accelerate kernel functions is usually to use shared memory, but certain specific data structures and access patterns can cause bank conflicts: <https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900>
        2. There are two ways to avoid bank conflicts: padding and swizzle. Padding is a very simple technique that adds some extra data at the end of a "row" to make the data in the same bank shift away, but in the same bank, if padding is not done properly, it may introduce miss alignment.
        3. Swizzle is a more complex technique, but it does not cause bank conflicts and allows you to perform vectorized memory access.
    Conditions for swizzle/bank conflict:
        1. The shape of the shared memory is T[array[-]][N], and N*sizeof(T) is a multiple of 32
        2. There is an access pattern of [tx][-] or one that leads to a stride of 128 bytes
        3. the above occurs in a warp thread (lds.32)
            1. if each thread in the warp accesses 4byte [lds.32], The request will not be split. It is necessary to determine whether there is a conflict in every 1/1 warp
            2. if each thread in warp accesses 8byte [lds.64], The request will be split into two times. It is necessary to determine whether there is a conflict in every 1/2 warp
            3. if each thread in the warp accesses 16byte [lds.128], The request will be split into 4 times. It is necessary to determine whether there is a conflict in every 1/4 warp
    How to perform swizzle: [y][x] -> [y][x_swz]
        1. swizzle_size = N * sizeof(T)
        2. access granularity is sizeof(TC) byte
        3. i_chunk = (y*N + x)*sizeof(T) / sizeof(TC)
        4. x_chunk = i_chunk % (swizzle_size / sizeof(TC)) , y_chunk = i_chunk / (swizzle_size / sizeof(TC))
        5. x_chunk_swz = y_chunk ^ x_chunk
        6. x_swz = x_chunk_swz * sizeof(TC) / sizeof(T) % N + x % (sizeof(TC) / sizeof(T))

    ``` shell
        nvcc 00_transpose.cu -o transpose
        ./transpose

        FP32 8192 x 8192
        Transpose with Shared Memory Bank Conflict: 0.71 ms
        Transpose without Shared Memory Bank Conflict via Padding: 0.43 ms
        Transpose without Shared Memory Bank Conflict via Swizzling: 0.43 ms
        FP16 8192 x 8192
        Transpose with Shared Memory Bank Conflict: 0.50 ms
        Transpose without Shared Memory Bank Conflict via Padding: 0.37 ms
        Transpose without Shared Memory Bank Conflict via Swizzling: 0.37 ms
        (H100 test result)
    ```

    Reference: https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
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

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

#define CDIV(M, N) (((M) + (N) - 1) / (N))

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
bool verify_transpose_implementation(
    std::function<void(T *, T const *, size_t, size_t, cudaStream_t)> transpose_function,
    size_t M, size_t N)
{
    // Fixed random seed for reproducibility
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const matrix_size = M * N;
    std::vector<T> matrix(matrix_size, 0.0f);
    std::vector<T> matrix_transposed(matrix_size, 1.0f);
    std::vector<T> matrix_transposed_reference(matrix_size, 2.0f);
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i = 0; i < matrix_size; ++i)
    {
        matrix[i] = uniform_dist(gen);
    }
    // Create the reference transposed matrix using CPU.
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            matrix_transposed_reference[j * M + i] = matrix[i * N + j];
        }
    }
    T *d_matrix;
    T *d_matrix_transposed;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), matrix_size * sizeof(T), cudaMemcpyHostToDevice));

    transpose_function(d_matrix_transposed, d_matrix, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(matrix_transposed.data(), d_matrix_transposed, matrix_size * sizeof(T), cudaMemcpyDeviceToHost));
    bool const correctness{is_equal(matrix_transposed.data(), matrix_transposed_reference.data(), matrix_size)};
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_matrix_transposed));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return correctness;
}

template <typename T>
float profile_transpose_implementation(std::function<void(T *, T const *, size_t, size_t, cudaStream_t)> transpose_function,
                                       size_t M, size_t N)
{
    constexpr int num_repeats = 100;
    constexpr int num_warmups = 10;
    cudaStream_t stream;
    size_t const matrix_size = M * N;
    T *d_matrix;
    T *d_matrix_transposed;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const wrapped_function = std::bind(transpose_function,
                                                                         d_matrix_transposed,
                                                                         d_matrix,
                                                                         M, N,
                                                                         std::placeholders::_1);
    float const function_latency = measure_performance(wrapped_function, stream, num_repeats, num_warmups);
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_matrix_transposed));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return function_latency;
}

void print_latencty(std::string const &kernel_name, float latency)
{
    std::cout << kernel_name << ": " << std::fixed << std::setprecision(2)
              << latency << " ms" << std::endl;
}

template <typename T>
void performance_test(const int M, const int N)
{
    float const latency_with_shm_bank_conflict =
        profile_transpose_implementation<T>(
            &launch_transpose_with_shm_bank_conflict<T>, M, N);
    print_latencty("Transpose with Shared Memory Bank Conflict",
                   latency_with_shm_bank_conflict);

    float const latency_without_shm_bank_conflict_via_padding =
        profile_transpose_implementation<T>(
            &launch_transpose_without_shm_bank_conflict_via_padding<T>, M, N);
    print_latencty("Transpose without Shared Memory Bank Conflict via Padding",
                   latency_without_shm_bank_conflict_via_padding);

    float const latency_without_shm_bank_conflict_via_swizzling =
        profile_transpose_implementation<T>(
            &launch_transpose_without_shm_bank_conflict_via_swizzling<T>, M, N);
    print_latencty("Transpose without Shared Memory Bank Conflict via Swizzling",
                   latency_without_shm_bank_conflict_via_swizzling);
}

template <typename T, size_t BLOCK_TILE_SIZE_X = 32, size_t BLOCK_TILE_SIZE_Y = 32, size_t PADDING_SIZE = 0>
__global__ void transpose(T *output_matrix, T const *input_matrix, size_t M, size_t N)
{
    // the input matrix is M*N, row major

    // Waste some shared memory to avoid bank conflicts if
    // PADDING_SIZE != 0.
    __shared__ T shm[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_X + PADDING_SIZE];

    // In some algorithms, such as matrix multiplication,
    // a warp of threads have to access a column of the 2D matrix in the shared
    // memory. Using the conventional index mapping, if the column size is not a
    // multiple of the warp size, there will be bank conflicts.
    size_t const input_matrix_x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t const input_matrix_y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t const input_matrix_idx = input_matrix_x + input_matrix_y * N;

    size_t shm_to_x = threadIdx.x;
    size_t shm_to_y = threadIdx.y;

    if ((input_matrix_y < M) && (input_matrix_x < N))
    {
        // Coalesced global memory access.
        shm[shm_to_y][shm_to_x] = input_matrix[input_matrix_idx];
    }

    // Make sure the buffer in a block is filled.
    __syncthreads();

    size_t const thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    // arange column major order
    size_t const shm_from_x = thread_idx / BLOCK_TILE_SIZE_Y;
    size_t const shm_from_y = thread_idx % BLOCK_TILE_SIZE_Y; // Coalesced direction

    size_t const output_matrix_x = shm_from_y + blockIdx.y * blockDim.y;
    size_t const output_matrix_y = shm_from_x + blockIdx.x * blockDim.x;
    size_t const output_matrix_idx = output_matrix_x + output_matrix_y * M;

    if ((output_matrix_y < N) && (output_matrix_x < M))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict if PADDING_SIZE = 1.
        output_matrix[output_matrix_idx] = shm[shm_from_y][shm_from_x];
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X = 32, size_t BLOCK_TILE_SIZE_Y = 32>
__global__ void transpose_swizzling(T *output_matrix, T const *input_matrix, size_t M, size_t N)
{
    __shared__ T shm[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_X];

    size_t const input_matrix_x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t const input_matrix_y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t const input_matrix_idx = input_matrix_x + input_matrix_y * N;
    size_t const shm_to_x = threadIdx.x;
    size_t const shm_to_y = threadIdx.y;
    size_t const shm_to_x_swizzled = (shm_to_x ^ shm_to_y) % BLOCK_TILE_SIZE_X;

    if ((input_matrix_y < M) && (input_matrix_x < N))
    {
        // Coalesced global memory access.
        shm[shm_to_y][shm_to_x_swizzled] = input_matrix[input_matrix_idx];
    }

    // Make sure the buffer in a block is filled.
    __syncthreads();

    size_t const block_thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    size_t const shm_from_x = block_thread_idx / BLOCK_TILE_SIZE_Y;
    size_t const shm_from_y = block_thread_idx % BLOCK_TILE_SIZE_Y;

    size_t const shm_from_x_swizzled = (shm_from_x ^ shm_from_y) % BLOCK_TILE_SIZE_X;
    size_t const output_matrix_x = shm_from_y + blockIdx.y * blockDim.y;
    size_t const output_matrix_y = shm_from_x + blockIdx.x * blockDim.x;
    size_t const output_matrix_idx = output_matrix_x + output_matrix_y * M;

    if ((output_matrix_y < N) && (output_matrix_x < M))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict.
        output_matrix[output_matrix_idx] = shm[shm_from_y][shm_from_x_swizzled];
    }
}

template <typename T>
void launch_transpose_with_shm_bank_conflict(T *d_output_matrix,
                                             T const *d_input_matrix, size_t M,
                                             size_t N, cudaStream_t stream)
{
    constexpr size_t BLOCK_TILE_SIZE_X = 32;
    constexpr size_t BLOCK_TILE_SIZE_Y = 32;
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X = 0;
    const dim3 block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y};
    const dim3 grid_size{static_cast<unsigned int>(CDIV(N, block_size.x)),
                         static_cast<unsigned int>(CDIV(M, block_size.y))};
    transpose<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SKEW_SIZE_X>
        <<<grid_size, block_size, 0, stream>>>(d_output_matrix, d_input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_transpose_without_shm_bank_conflict_via_padding(
    T *d_output_matrix, T const *d_input_matrix, size_t M, size_t N,
    cudaStream_t stream)
{
    constexpr size_t BLOCK_TILE_SIZE_X{32};
    constexpr size_t BLOCK_TILE_SIZE_Y{32};
    // if you use other datatype or vetorized memory access
    // skew size can be different
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{1};
    dim3 const block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y};
    dim3 const grid_size{static_cast<unsigned int>(CDIV(N, block_size.x)),
                         static_cast<unsigned int>(CDIV(M, block_size.y))};
    transpose<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SKEW_SIZE_X>
        <<<grid_size, block_size, 0, stream>>>(d_output_matrix, d_input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_transpose_without_shm_bank_conflict_via_swizzling(
    T *d_output_matrix, T const *d_input_matrix, size_t M, size_t N,
    cudaStream_t stream)
{
    constexpr size_t BLOCK_TILE_SIZE_X = 32;
    constexpr size_t BLOCK_TILE_SIZE_Y = 32;
    dim3 const block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y};
    dim3 const grid_size{static_cast<unsigned int>(CDIV(N, block_size.x)),
                         static_cast<unsigned int>(CDIV(M, block_size.y))};
    transpose_swizzling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y><<<grid_size, block_size, 0, stream>>>(
        d_output_matrix, d_input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
bool is_equal(T const *data_1, T const *data_2, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data_1[i] != data_2[i])
        {
            return false;
        }
    }
    return true;
}

template <typename T>
void unit_test()
{
    for (size_t m = 1; m <= 33; ++m)
    {
        for (size_t n = 1; n <= 33; ++n)
        {
            assert(verify_transpose_implementation<T>(
                &launch_transpose_with_shm_bank_conflict<T>, m, n));

            assert(verify_transpose_implementation<T>(
                &launch_transpose_without_shm_bank_conflict_via_padding<T>, m, n));

            assert(verify_transpose_implementation<T>(
                &launch_transpose_without_shm_bank_conflict_via_swizzling<T>, m, n));
        }
    }
}

int main()
{
    unit_test<float>();
    size_t const M = 8192;
    size_t const N = 8192;
    std::cout << "FP32 " << M << " x " << N << std::endl;
    performance_test<float>(M, N);

    unit_test<half>();
    std::cout << "FP16 " << M << " x " << N << std::endl;
    performance_test<half>(M, N);
    return 0;
}