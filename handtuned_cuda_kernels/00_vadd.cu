/*
    simple vector add kernel
    nvcc 00_vadd.cu -o vadd
    ./vadd
*/
#include <iomanip>
#include <iostream>
#include <functional>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define HALF4(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
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

template <typename T>
__global__ void vadd(T *a, T *b, T *c, size_t N)
{
    size_t off = blockDim.x * blockIdx.x + threadIdx.x;
    if (off >= N)
        return;
    c[off] = a[off] + b[off];
}

template <typename T>
__forceinline__ void launch_vadd(T *a, T *b, T *c, size_t N, cudaStream_t stream)
{
    dim3 block_size = {128};
    dim3 grid_size = {static_cast<unsigned int>((N + 127) / 128)};
    vadd<T><<<grid_size, block_size>>>(a, b, c, N);
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
float vadd_benchmark(std::function<void(T *, T *, T *, size_t, cudaStream_t)> _function,
                     size_t N)
{
    constexpr int num_repeats = 10;
    constexpr int num_warmups = 20;
    cudaStream_t stream;
    T *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(T)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const wrapped_function = std::bind(_function,
                                                                         d_a,
                                                                         d_b,
                                                                         d_c,
                                                                         N,
                                                                         std::placeholders::_1);
    float const function_latency = measure_performance(wrapped_function, stream, num_repeats, num_warmups);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return function_latency;
}

void print_latencty(std::string const &kernel_name, float latency)
{
    std::cout << kernel_name << ": " << std::fixed << std::setprecision(3)
              << latency << " ms" << std::endl;
}

template <typename T>
void performance_test(const int N)
{
    float const latency_with_shm_bank_conflict = vadd_benchmark<T>(&launch_vadd<T>, N);
    print_latencty("Vector Addition", latency_with_shm_bank_conflict);
}

int main()
{
    const int N = 16384;
    performance_test<float>(N);
    return 0;
}