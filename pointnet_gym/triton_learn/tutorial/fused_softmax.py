import torch

import triton
import triton.language as tl
from triton.runtime import driver


def is_hip():
    """
    AMD的一种类CUDA编程接口,可以跨平台
    """
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    """
    AMD GPU的一种架构
    """
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx90a",
        "gfx908",
    )


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # starting row of the program
    """
    数据的形状是M*N
    """
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        """
        BLOCK_SIZE的大小是2的幂,且大于n_cols,一个block可以处理一行
        """
        col_offsets = tl.arange(0, BLOCK_SIZE)
        """
        对于block中的每一个线程,还需要加上相应的偏移
        """
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        """
        如果超出N,则mask为False,返回other
        """
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        """
        @triton.jit 会自动融合下面的操作,避免中间量多次存取
        """
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


torch.cuda.set_device(2)  # 选择设备
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)  # 获取设备参数
NUM_SM = properties["multiprocessor_count"]  # SM数量
NUM_REGS = properties["max_num_regs"]  # 每个SM上的寄存器数量
SIZE_SMEM = properties["max_shared_mem"]  # 每个SM上的共享内存大小
WARP_SIZE = properties["warpSize"]  # 每个warp的线程数
# target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        kernel._init_handles()
        """
        kernel.n_regs应该是一个线程需要的寄存器数
        """
        n_regs = max(1, kernel.n_regs)
        size_smem = max(1, kernel.metadata.shared)
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            """
            VGPRs(矢量通用寄存器)
            """
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            """
            waves就对应CUDA中的一个sm上最多能并行多少个warp
            """
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            """
            这里occypancy表示一个处理器上能并行的block数
            """
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)
    """
    对于nums_programs个block,每个block处理一行
    当数据的rows很多的时候,每个block串行的完成rows/num_programs行
    """
    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    """
    这里相当于就已经创建了一个实例
    """
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y


def softmax_test():
    torch.manual_seed(0)
    x = torch.randn(1024, 10, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def softmax_benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    softmax_test()
    # softmax_benchmark.run(print_data=True, show_plots=False, save_path="./out")
