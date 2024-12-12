# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习推理过程，请严格保持输出格式输出
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import h5py
import time
from typing import Dict, List, Set

import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.runtime import driver

import numpy as np

DTYPE = torch.float16
NPTYPE = np.float16
TL_TYPE = tl.float16 if DTYPE == torch.float16 else tl.float32

DBG_FLAG = True


def get_cuda_autotune_config(dtype: torch.dtype):
    if dtype == torch.float16:
        return [
            # Good config for fp8 inputs.triton.Config
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 256,
            #         "BLOCK_SIZE_K": 128,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=3,
            #     num_warps=8,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 256,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 128,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=3,
            #     num_warps=8,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 256,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 128,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 256,
            #         "BLOCK_SIZE_K": 128,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 128,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 64,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 64,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
        ]
    else:
        return [
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 16,
            #         "GROUP_SIZE_M": 1,
            #     },
            #     num_stages=3,
            #     num_warps=8,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 256,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 16,
            #         "GROUP_SIZE_M": 2,
            #     },
            #     num_stages=3,
            #     num_warps=8,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 256,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 16,
            #         "GROUP_SIZE_M": 2,
            #     },
            #     num_stages=3,
            #     num_warps=8,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 256,
            #         "BLOCK_SIZE_K": 64,
            #         "GROUP_SIZE_M": 4,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 256,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 4,
                },
                num_stages=4,
                num_warps=4,
            ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 4,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 32,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 4,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 32,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=5,
            #     num_warps=2,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 32,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=5,
            #     num_warps=2,
            # ),
        ]


# We can fuse `relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0.0)


@triton.autotune(
    configs=get_cuda_autotune_config(DTYPE),
    key=["M", "N", "K"],
)
@triton.jit
def biased_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    v_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """
    Kernel for computing the matmul C = A x B + v.
    A has shape (M, K), B has shape (K, N), C has shape (M, N) and v has shape (N,).

    Map program ids `pid` to the block of C it should compute.
    每个block负责计算C的一个BLOCK_SIZE_M*BLOCK_SIZE_N的子矩阵

    This is done in a grouped ordering to promote L2 data reuse.
    See above `L2 Cache Optimizations` section for details.

    每个group
    +------------   N  -------------|
    |--BN --|
    +-------------------------------+  ——   ————
    |       |       |       |       |  B     |
    |       |       |       |       |  M     |
    +-------------------------------+  ——    |
    |       |       |       |       |        GROUP_SIZE_M
    |       |       |       |       |        |
    +-------------------------------+        |
    |       |       |       |       |        |
    |       |       |       |       |        |
    +-------------------------------+      ————
    有GROUP_SIZE_M * num_pid_n个block
    这些block的排列方式是列主序的(L2 Cache Optimizations)
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    """
    num_pid_m就是表示行方向上有多少个block
    num_pid_n就是表示列方向上有多少个block
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    """
    处理最后一些行方向上不满GROUP_SIZE_M
    """
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    """
    group内的block划分
    """
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # load bias
    v_ptrs = v_ptr + offs_bn
    v = tl.load(v_ptrs)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator += v

    # You can fuse arbitrary activation functions here
    if ACTIVATION == "relu":
        accumulator = relu(accumulator)

    # while the accumulator is still in FP32!
    c = accumulator.to(TL_TYPE)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def biased_gemm_caller(
    M,
    N,
    K,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    v: torch.Tensor,
    activation="",
):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    """
    张量的stride
    tensor a.shape=(x,y,z)
    a.stride = (y*z, z, 1)
    """
    biased_gemm_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        v_ptr=v,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        ACTIVATION=activation,
    )
    return c


def biased_gemm_fuse_addbasis_caller(
    M,
    N,
    K,
    r: int,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    v: torch.Tensor,
    activation="",
):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    """
    张量的stride
    tensor a.shape=(x,y,z)
    a.stride = (y*z, z, 1)
    """
    idenm = torch.eye(r, device=a.device, dtype=DTYPE).reshape(-1)
    v = v + idenm
    biased_gemm_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        v_ptr=v,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        ACTIVATION=activation,
    )
    return c


@triton.autotune(
    configs=get_cuda_autotune_config(DTYPE),
    key=["M", "N", "K"],
)
@triton.jit
def biased_gemm_fuse_pbn_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    v_ptr,
    alpha_ptr,
    beta_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """
    alpha, beta is preprocess bn layer
    Kernel for computing the matmul C = (A x B + v) x alpha + beta.
    A has shape (M, K), B has shape (K, N), C has shape (M, N) and v has shape (N,).

    Map program ids `pid` to the block of C it should compute.
    每个block负责计算C的一个BLOCK_SIZE_M*BLOCK_SIZE_N的子矩阵

    This is done in a grouped ordering to promote L2 data reuse.
    See above `L2 Cache Optimizations` section for details.

    每个group
    +------------   N  -------------|
    |--BN --|
    +-------------------------------+  ——   ————
    |       |       |       |       |  B     |
    |       |       |       |       |  M     |
    +-------------------------------+  ——    |
    |       |       |       |       |        GROUP_SIZE_M
    |       |       |       |       |        |
    +-------------------------------+        |
    |       |       |       |       |        |
    |       |       |       |       |        |
    +-------------------------------+      ————
    有GROUP_SIZE_M * num_pid_n个block
    这些block的排列方式是列主序的(L2 Cache Optimizations)
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    """
    num_pid_m就是表示行方向上有多少个block
    num_pid_n就是表示列方向上有多少个block
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    """
    处理最后一些行方向上不满GROUP_SIZE_M
    """
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    """
    group内的block划分
    """
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # load bias
    v_ptrs = v_ptr + offs_bn
    v = tl.load(v_ptrs)

    # load bn preprocessed param
    alpha_ptrs = alpha_ptr + offs_bn
    alpha = tl.load(alpha_ptrs)
    beta_ptrs = beta_ptr + offs_bn
    beta = tl.load(beta_ptrs)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator += v

    # fusion bn
    accumulator *= alpha
    accumulator += beta

    # You can fuse arbitrary activation functions here
    if ACTIVATION == "relu":
        accumulator = relu(accumulator)

    # while the accumulator is still in FP32!
    c = accumulator.to(TL_TYPE)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def biased_gemm_fuse_pbn_caller(
    M,
    N,
    K,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    activation="",
    # BLOCK_SIZE_M=16,
    # BLOCK_SIZE_N=16,
    # BLOCK_SIZE_K=16,
    # GROUP_SIZE_M=1,
):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    """
    张量的stride
    tensor a.shape=(x,y,z)
    a.stride = (y*z, z, 1)
    """
    biased_gemm_fuse_pbn_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        v_ptr=v,
        alpha_ptr=alpha,
        beta_ptr=beta,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        ACTIVATION=activation,
        # BLOCK_SIZE_M=BLOCK_SIZE_M,
        # BLOCK_SIZE_N=BLOCK_SIZE_N,
        # BLOCK_SIZE_K=BLOCK_SIZE_K,
        # GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c


@triton.autotune(
    configs=get_cuda_autotune_config(DTYPE),
    key=["M", "N", "K"],
)
@triton.jit
def batched_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """
    alpha, beta is preprocess bn layer
    Kernel for computing the matmul C = (A x B + v) x alpha + beta.
    A has shape (M, K), B has shape (K, N), C has shape (M, N) and v has shape (N,).

    Map program ids `pid` to the block of C it should compute.
    每个block负责计算C的一个BLOCK_SIZE_M*BLOCK_SIZE_N的子矩阵

    This is done in a grouped ordering to promote L2 data reuse.
    See above `L2 Cache Optimizations` section for details.

    每个group
    +------------   N  -------------|
    |--BN --|
    +-------------------------------+  ——   ————
    |       |       |       |       |  B     |
    |       |       |       |       |  M     |
    +-------------------------------+  ——    |
    |       |       |       |       |        GROUP_SIZE_M
    |       |       |       |       |        |
    +-------------------------------+        |
    |       |       |       |       |        |
    |       |       |       |       |        |
    +-------------------------------+      ————
    有GROUP_SIZE_M * num_pid_n个block
    这些block的排列方式是列主序的(L2 Cache Optimizations)
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    batch_idx = tl.program_id(1).to(tl.int64)
    """
    num_pid_m就是表示行方向上有多少个block
    num_pid_n就是表示列方向上有多少个block
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    """
    处理最后一些行方向上不满GROUP_SIZE_M
    """
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    """
    group内的block划分
    """
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    a_batch_base_offset = batch_idx * M * K
    b_batch_base_offset = batch_idx * K * N
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = (
        a_ptr
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        + a_batch_base_offset
    )
    b_ptrs = (
        b_ptr
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        + b_batch_base_offset
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    if ACTIVATION == "relu":
        accumulator = relu(accumulator)

    # while the accumulator is still in FP32!
    c = accumulator.to(TL_TYPE)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_batch_base_offset = batch_idx * M * N
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
        + c_batch_base_offset
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def batched_gemm_caller(
    batch_size,
    M,
    N,
    K,
    a: torch.Tensor,  # [bs * m, k]
    b: torch.Tensor,  # [bs * k, n]
    c: torch.Tensor,  # [bs * m, n]
    activation="",
):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        batch_size,
    )
    """
    张量的stride
    tensor a.shape=(x,y,z)
    a.stride = (y*z, z, 1)
    """
    batched_gemm_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        K=K,
        stride_am=K,
        stride_ak=1,
        stride_bk=N,
        stride_bn=1,
        stride_cm=K,
        stride_cn=1,
        ACTIVATION=activation,
        # BLOCK_SIZE_M=128,
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=64,
        # GROUP_SIZE_M=8,
        # num_warps=4,
        # num_ctas=1,
        # num_stages=4,
    )
    return c


@triton.autotune(
    configs=[
        # triton.Config({}, num_stages=1, num_warps=2),
        # triton.Config({}, num_stages=2, num_warps=4),
        # triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=16),
    ],
    key=["BLOCK_SIZE_N", "BLOCK_SIZE_C"],
)
@triton.jit
def max_dim1_kernel(
    a_ptr,
    out_ptr,
    n_rows: int,
    n_cols: int,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    num_stage: tl.constexpr,
):
    """
    This kernel computes the maximum value of a tensor at dim 1
    | bc|
    +-------------------+   ——      ——
    |   |   |   |   |   |   |       |
    |   |   |   |   |   |   bn      |
    |   |   |   |   |   |   |       |
    |   |   |   |   |   |   ——      |
    |   |   |   |   |   |   |       |
    |   |   |   |   |   |   bn      batch
    |   |   |   |   |   |   |       |
    |   |   |   |   |   |   ——      |
    |   |   |   |   |   |           |
    |   |   |   |   |   |           |
    +-------------------+           ——

    grid = (cdiv(n_cols, bc), batch_size)
    每个block处理一个batch的BLOCK_SIZE_C列,分成cdiv(n_rows, BLOCK_SIZE_N)次循环处理完成
    """
    batch_idx = tl.program_id(1).to(tl.int64)
    """
    不知道tl.program_id为什么仅仅使用int32
    /opt/conda/envs/py312/lib/python3.12/site-packages/triton/language/semantic.py:26
    batch_idx * n_rows * n_cols 太大了会溢出(而且不会警告),要手动强制转换
    /opt/conda/envs/py312/lib/python3.12/site-packages/triton/runtime/jit.py:311
    """
    batch_base_off = batch_idx * n_rows * n_cols
    row_step = BLOCK_SIZE_N
    col_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE_C

    accumulator = tl.full((BLOCK_SIZE_C,), -float("inf"), dtype=tl.float32)
    for row_idx in tl.range(0, n_rows, row_step, num_stages=num_stage):
        row_offsets = row_idx + tl.arange(0, BLOCK_SIZE_N)
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_C)
        a_ptrs = (
            a_ptr
            + batch_base_off
            + row_offsets[:, None] * n_cols
            + col_offsets[None, :]
        )
        a = tl.load(
            a_ptrs,
            mask=(row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols),
            other=-float("inf"),
        )
        blk_max = tl.max(a, axis=0)
        accumulator = tl.where(blk_max > accumulator, blk_max, accumulator)

    out_ptrs = out_ptr + batch_idx * n_cols + col_start + tl.arange(0, BLOCK_SIZE_C)
    tl.store(
        out_ptrs, accumulator, mask=tl.arange(0, BLOCK_SIZE_C) < n_cols - col_start
    )


def max_dim1_caller(batch_size, n, c, x: torch.Tensor, out: torch.Tensor):
    assert x.dtype == DTYPE, "Matrix A must be of dtype DTYPE"
    assert out.dtype == DTYPE, "Matrix out must be of dtype DTYPE"
    BLOCK_SIZE_N = 1024
    BLOCK_SIZE_C = min(triton.next_power_of_2(c), 64)
    grid = (triton.cdiv(c, BLOCK_SIZE_C), batch_size)
    max_dim1_kernel[grid](
        a_ptr=x,
        out_ptr=out,
        n_rows=n,
        n_cols=c,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_stage=2,
    )
    return out


@triton.jit
def logsoftmax_dim1_kernel(
    input_ptr,
    output_ptr,
    n_rows: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):

    row_start = tl.program_id(0).to(tl.int64)
    row_step = tl.num_programs(0).to(tl.int64)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * n_cols
        """
        BLOCK_SIZE是最小的大于等于n_cols的2的幂
        """
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        denominator = tl.log(tl.sum(tl.exp(row_minus_max), axis=0) + 1e-8)
        logsoftmax_out = row_minus_max - denominator
        output_row_start_ptr = output_ptr + row_idx * n_cols
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, logsoftmax_out, mask=col_offsets < n_cols)


logsoftmax_kernels = {}


def logsoftmax_dim1_warmup(x: torch.Tensor):
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)  # 获取设备参数
    NUM_SM = properties["multiprocessor_count"]  # SM数量
    NUM_REGS = properties["max_num_regs"]  # 每个SM上的寄存器数量
    SIZE_SMEM = properties["max_shared_mem"]  # 每个SM上的共享内存大小
    WARP_SIZE = properties["warpSize"]  # 每个warp的线程数
    n_rows, n_cols = x.shape
    global logsoftmax_kernels
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Allocate output
    y = torch.empty_like(x)
    num_warps = 8
    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = logsoftmax_dim1_kernel.warmup(
        x,
        y,
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
    """
    这里occypancy表示一个处理器上能并行的block数
    """
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    logsoftmax_kernels[BLOCK_SIZE] = (kernel, num_programs)


def logsoftmax_dim1_caller(x: torch.Tensor, out: torch.Tensor):
    global logsoftmax_kernels
    if len(logsoftmax_kernels) == 0:
        logsoftmax_dim1_warmup(x)
    n_rows, n_cols = x.shape
    """
    对于nums_programs个block,每个block处理一行
    当数据的rows很多的时候,每个block串行的完成rows/num_programs行
    """
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    kernel, num_programs = logsoftmax_kernels.get(BLOCK_SIZE, (None, 0))
    num_programs = min(num_programs, n_rows)
    # Create a number of persistent programs.
    """
    这里相当于就已经创建了一个实例
    """
    kernel[(num_programs, 1, 1)](
        x,
        out,
        n_rows,
        n_cols,
    )
    return out


@triton.jit
def preprocess_bn_params_kernel(
    w_ptr, b_ptr, mean_ptr, var_ptr, N, eps, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    col_off = pid * BLOCK_SIZE
    w_ptrs = w_ptr + col_off + tl.arange(0, BLOCK_SIZE)
    b_ptrs = b_ptr + col_off + tl.arange(0, BLOCK_SIZE)
    mean_ptrs = mean_ptr + col_off + tl.arange(0, BLOCK_SIZE)
    var_ptrs = var_ptr + col_off + tl.arange(0, BLOCK_SIZE)
    w = tl.load(w_ptrs, mask=col_off < N)
    b = tl.load(b_ptrs, mask=col_off < N)
    mean = tl.load(mean_ptrs, mask=col_off < N)
    var = tl.load(var_ptrs, mask=col_off < N)

    denominator = tl.rsqrt(var + eps)
    w = w * denominator
    b = b - mean * w

    tl.store(w_ptrs, w, mask=col_off < N)
    tl.store(b_ptrs, b, mask=col_off < N)


@triton.jit
def argmax_dim1_kernel(
    a_ptr,
    out_ptr,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    a_ptr = a_ptr + pid * n_cols
    cur_max = -float("inf")
    arg_max = 0
    arg_max = arg_max.to(tl.int32)
    for idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        col_off = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a_ptrs = a_ptr + col_off
        a = tl.load(a_ptrs, mask=col_off < n_cols, other=-float("inf"))
        tmp_max = tl.max(a, axis=0)
        if tmp_max > cur_max:
            cur_max = tmp_max
            arg_max = tl.argmax(a, axis=0) + idx * BLOCK_SIZE
    out_ptr = out_ptr + pid
    tl.store(out_ptr, arg_max)


def argmax_dim1_caller(x: torch.Tensor, out: torch.Tensor):
    assert x.is_contiguous(), "Matrix A must be contiguous"
    assert x.dtype == DTYPE, "Matrix A must be of dtype DTYPE"
    n, n_cols = x.shape
    assert n == out.shape[0], "Incompatible dimensions"
    BLOCK_SIZE = 16
    grid = (n,)
    argmax_dim1_kernel[grid](
        a_ptr=x,
        out_ptr=out,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=2,
    )
    return out


wt_param_name: Dict[str, List[int]] = {
    "feat.stn.conv1.weight": [64, 3],
    "feat.stn.conv2.weight": [128, 64],
    "feat.stn.conv3.weight": [1024, 128],
    "feat.stn.fc1.weight": [512, 1024],
    "feat.stn.fc2.weight": [256, 512],
    "feat.stn.fc3.weight": [9, 256],
    "feat.conv1.weight": [64, 3],
    "feat.fstn.conv1.weight": [64, 64],
    "feat.fstn.conv2.weight": [128, 64],
    "feat.fstn.conv3.weight": [1024, 128],
    "feat.fstn.fc1.weight": [512, 1024],
    "feat.fstn.fc2.weight": [256, 512],
    "feat.fstn.fc3.weight": [4096, 256],
    "feat.conv2.weight": [128, 64],
    "feat.conv3.weight": [1024, 128],
    "fc1.weight": [512, 1024],
    "fc2.weight": [256, 512],
    "fc3.weight": [10, 256],
}

param_name: Dict[str, List[int]] = {
    "feat.stn.conv1.weight": [64, 3],
    "feat.stn.conv1.bias": [64],
    "feat.stn.conv2.weight": [128, 64],
    "feat.stn.conv2.bias": [128],
    "feat.stn.conv3.weight": [1024, 128],
    "feat.stn.conv3.bias": [1024],
    "feat.stn.fc1.weight": [512, 1024],
    "feat.stn.fc1.bias": [512],
    "feat.stn.fc2.weight": [256, 512],
    "feat.stn.fc2.bias": [256],
    "feat.stn.fc3.weight": [9, 256],
    "feat.stn.fc3.bias": [9],
    "feat.stn.bn1.weight": [64],
    "feat.stn.bn1.bias": [64],
    "feat.stn.bn1.running_mean": [64],
    "feat.stn.bn1.running_var": [64],
    "feat.stn.bn2.weight": [128],
    "feat.stn.bn2.bias": [128],
    "feat.stn.bn2.running_mean": [128],
    "feat.stn.bn2.running_var": [128],
    "feat.stn.bn3.weight": [1024],
    "feat.stn.bn3.bias": [1024],
    "feat.stn.bn3.running_mean": [1024],
    "feat.stn.bn3.running_var": [1024],
    "feat.stn.bn4.weight": [512],
    "feat.stn.bn4.bias": [512],
    "feat.stn.bn4.running_mean": [512],
    "feat.stn.bn4.running_var": [512],
    "feat.stn.bn5.weight": [256],
    "feat.stn.bn5.bias": [256],
    "feat.stn.bn5.running_mean": [256],
    "feat.stn.bn5.running_var": [256],
    "feat.conv1.weight": [64, 3],
    "feat.conv1.bias": [64],
    "feat.bn1.weight": [64],
    "feat.bn1.bias": [64],
    "feat.bn1.running_mean": [64],
    "feat.bn1.running_var": [64],
    "feat.conv2.weight": [128, 64],
    "feat.conv2.bias": [128],
    "feat.bn2.weight": [128],
    "feat.bn2.bias": [128],
    "feat.bn2.running_mean": [128],
    "feat.bn2.running_var": [128],
    "feat.conv3.weight": [1024, 128],
    "feat.conv3.bias": [1024],
    "feat.bn3.weight": [1024],
    "feat.bn3.bias": [1024],
    "feat.bn3.running_mean": [1024],
    "feat.bn3.running_var": [1024],
    "feat.fstn.conv1.weight": [64, 64],
    "feat.fstn.conv1.bias": [64],
    "feat.fstn.conv2.weight": [128, 64],
    "feat.fstn.conv2.bias": [128],
    "feat.fstn.conv3.weight": [1024, 128],
    "feat.fstn.conv3.bias": [1024],
    "feat.fstn.fc1.weight": [512, 1024],
    "feat.fstn.fc1.bias": [512],
    "feat.fstn.fc2.weight": [256, 512],
    "feat.fstn.fc2.bias": [256],
    "feat.fstn.fc3.weight": [4096, 256],
    "feat.fstn.fc3.bias": [4096],
    "feat.fstn.bn1.weight": [64],
    "feat.fstn.bn1.bias": [64],
    "feat.fstn.bn1.running_mean": [64],
    "feat.fstn.bn1.running_var": [64],
    "feat.fstn.bn2.weight": [128],
    "feat.fstn.bn2.bias": [128],
    "feat.fstn.bn2.running_mean": [128],
    "feat.fstn.bn2.running_var": [128],
    "feat.fstn.bn3.weight": [1024],
    "feat.fstn.bn3.bias": [1024],
    "feat.fstn.bn3.running_mean": [1024],
    "feat.fstn.bn3.running_var": [1024],
    "feat.fstn.bn4.weight": [512],
    "feat.fstn.bn4.bias": [512],
    "feat.fstn.bn4.running_mean": [512],
    "feat.fstn.bn4.running_var": [512],
    "feat.fstn.bn5.weight": [256],
    "feat.fstn.bn5.bias": [256],
    "feat.fstn.bn5.running_mean": [256],
    "feat.fstn.bn5.running_var": [256],
    "fc1.weight": [512, 1024],
    "fc1.bias": [512],
    "bn1.weight": [512],
    "bn1.bias": [512],
    "bn1.running_mean": [512],
    "bn1.running_var": [512],
    "fc2.weight": [256, 512],
    "fc2.bias": [256],
    "bn2.weight": [256],
    "bn2.bias": [256],
    "bn2.running_mean": [256],
    "bn2.running_var": [256],
    "fc3.weight": [10, 256],
    "fc3.bias": [10],
}

bn_layer_name: Set[str] = [
    "feat.stn.bn1",
    "feat.stn.bn2",
    "feat.stn.bn3",
    "feat.stn.bn4",
    "feat.stn.bn5",
    "feat.bn1",
    "feat.bn2",
    "feat.bn3",
    "feat.fstn.bn1",
    "feat.fstn.bn2",
    "feat.fstn.bn3",
    "feat.fstn.bn4",
    "feat.fstn.bn5",
    "bn1",
    "bn2",
]

layer_out_shape: Dict[str, List[int]] = None


def set_layer_out_shape(batch_size, n, total_points):
    global layer_out_shape
    layer_out_shape = {
        "feat.stn.conv1": [batch_size * n, 64],
        "feat.stn.conv2": [batch_size * n, 128],
        "feat.stn.conv3": [batch_size * n, 1024],
        "feat.stn.maxpool": [batch_size, 1024],
        "feat.stn.fc1": [batch_size, 512],
        "feat.stn.fc2": [batch_size, 256],
        "feat.stn.fc3": [batch_size, 9],  # add basis
        "feat.bmm1": [batch_size * n, 3],
        "feat.conv1": [batch_size * n, 64],
        "feat.fstn.conv1": [batch_size * n, 64],
        "feat.fstn.conv2": [batch_size * n, 128],
        "feat.fstn.conv3": [batch_size * n, 1024],
        "feat.fstn.maxpool": [batch_size, 1024],
        "feat.fstn.fc1": [batch_size, 512],
        "feat.fstn.fc2": [batch_size, 256],
        "feat.fstn.fc3": [batch_size, 4096],  # add basis
        "feat.bmm2": [batch_size * n, 64],
        "feat.conv2": [batch_size * n, 128],
        "feat.conv3": [batch_size * n, 1024],
        "feat.maxpool": [batch_size, 1024],
        "fc1": [batch_size, 512],
        "fc2": [batch_size, 256],
        "fc3": [total_points, 10],
    }


def read_params(dir):
    # 列出所有txt文件
    global param_name
    files = [f for f in os.listdir(dir) if f.endswith(".txt")]
    params = {}
    need_params_names = param_name.keys()
    for fileName in files:
        data = []
        modelName = fileName.replace(".txt", "")
        if modelName not in need_params_names:
            continue
        with open(os.path.join(dir, fileName), "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                value = float(line)
                data.append(value)
        params[modelName] = data
    return params


def preprocess_bn_params_caller(
    w: torch.Tensor, b: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
):
    n = w.shape[0]
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    preprocess_bn_params_kernel[grid](
        w_ptr=w,
        b_ptr=b,
        mean_ptr=mean,
        var_ptr=var,
        N=n,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def preprocess_params(
    params: Dict[str, torch.Tensor], type: torch.dtype = torch.float16
):
    for k in params.keys():
        t_v = torch.tensor(params[k], dtype=type).reshape(param_name[k]).cuda()
        if k in wt_param_name:
            t_v = t_v.transpose_(0, 1)
        params[k] = t_v
    # 将模型参数转换为tensor
    for k in bn_layer_name:
        w = params[k + ".weight"]
        b = params[k + ".bias"]
        mean = params[k + ".running_mean"]
        var = params[k + ".running_var"]
        preprocess_bn_params_caller(w, b, mean, var)
        params.pop(k + ".running_mean")
        params.pop(k + ".running_var")
    return params


def dataset_sample(points: np.ndarray, fix_length: int):
    # new_points = None
    # if points.shape[0] >= fix_length:
    #     new_points = points[:fix_length, :]
    # else:
    #     new_points = np.concatenate(
    #         (
    #             points,
    #             np.zeros((fix_length - points.shape[0], 3), dtype=np.float32),
    #         ),
    #         axis=0,
    #     )

    if points.shape[0] < fix_length:
        points = np.concatenate(
            (
                points,
                np.zeros((fix_length - points.shape[0], 3), dtype=np.float32),
            ),
            axis=0,
        )
    steps = points.shape[0] // fix_length
    return points[: steps * fix_length : steps,]


def read_h5_file(dataPath, fixed_length):
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath, "r") as hf:
        for k in hf.keys():
            list_of_points.append(
                dataset_sample(hf[k]["points"][:].astype(NPTYPE), fixed_length)
            )
            list_of_labels.append(hf[k].attrs["label"])
    return np.array(list_of_points), np.array(list_of_labels)


def alloc_layer_out_tensor_mem(dtype: torch.dtype):
    out_tensors: Dict[str, torch.Tensor] = {}
    global layer_out_shape
    for k in layer_out_shape.keys():
        out_tensors[k] = torch.empty(layer_out_shape[k], dtype=dtype, device="cuda")
    return out_tensors


def dump_np(input: np.ndarray, file: str, mode="w"):
    if len(input.shape) == 1:
        with open(file, mode) as f:
            for d in input:
                f.write(str(d) + " ")
            f.write("\n")
    else:
        if len(input.shape) > 2:
            input = input.reshape(-1, input.shape[-1])
        with open(file, mode) as f:
            for rows in input:
                f.write(" ".join(map(str, rows)) + "\n")
    exit(0)


def do_inference(
    list_of_points: np.ndarray,
    list_of_labels: torch.Tensor,
    params: Dict[str, torch.Tensor],
    batch_size: int,
    fixed_length: int,
    layer_out_mem: Dict[str, torch.Tensor],
    logsoftmax_out: torch.Tensor,
    inference_labels: torch.Tensor,
    dtype: torch.dtype,
):  # 请在本函数下使用triton实现推理操作
    dataset_size = list_of_points.shape[0]
    idx = 0
    while idx + batch_size <= dataset_size and batch_size > 0:
        batch_points = list_of_points[idx : idx + batch_size]
        ## 1. 将点云数据转换为tensor
        points = torch.tensor(batch_points, dtype=dtype).cuda().reshape(-1, 3)
        ## 2. 使用triton进行推理
        # feat.stn
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=64,
            K=3,
            a=points,
            b=params["feat.stn.conv1.weight"],
            c=layer_out_mem["feat.stn.conv1"],
            v=params["feat.stn.conv1.bias"],
            alpha=params["feat.stn.bn1.weight"],
            beta=params["feat.stn.bn1.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=128,
            K=64,
            a=layer_out_mem["feat.stn.conv1"],
            b=params["feat.stn.conv2.weight"],
            c=layer_out_mem["feat.stn.conv2"],
            v=params["feat.stn.conv2.bias"],
            alpha=params["feat.stn.bn2.weight"],
            beta=params["feat.stn.bn2.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=1024,
            K=128,
            a=layer_out_mem["feat.stn.conv2"],
            b=params["feat.stn.conv3.weight"],
            c=layer_out_mem["feat.stn.conv3"],
            v=params["feat.stn.conv3.bias"],
            alpha=params["feat.stn.bn3.weight"],
            beta=params["feat.stn.bn3.bias"],
            activation="relu",
        )
        max_dim1_caller(
            batch_size=batch_size,
            n=fixed_length,
            c=1024,
            x=layer_out_mem["feat.stn.conv3"],
            out=layer_out_mem["feat.stn.maxpool"],
        )
        # dump_np(layer_out_mem["feat.stn.maxpool"].cpu().numpy(), "triton.txt")
        biased_gemm_fuse_pbn_caller(
            M=batch_size,
            N=512,
            K=1024,
            a=layer_out_mem["feat.stn.maxpool"],
            b=params["feat.stn.fc1.weight"],
            c=layer_out_mem["feat.stn.fc1"],
            v=params["feat.stn.fc1.bias"],
            alpha=params["feat.stn.bn4.weight"],
            beta=params["feat.stn.bn4.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size,
            N=256,
            K=512,
            a=layer_out_mem["feat.stn.fc1"],
            b=params["feat.stn.fc2.weight"],
            c=layer_out_mem["feat.stn.fc2"],
            v=params["feat.stn.fc2.bias"],
            alpha=params["feat.stn.bn5.weight"],
            beta=params["feat.stn.bn5.bias"],
            activation="relu",
        )
        biased_gemm_fuse_addbasis_caller(
            M=batch_size,
            N=9,
            K=256,
            r=3,
            a=layer_out_mem["feat.stn.fc2"],
            b=params["feat.stn.fc3.weight"],
            c=layer_out_mem["feat.stn.fc3"],
            v=params["feat.stn.fc3.bias"],
        )
        # feat
        batched_gemm_caller(
            batch_size=batch_size,
            M=fixed_length,
            N=3,
            K=3,
            a=points,
            b=layer_out_mem["feat.stn.fc3"],
            c=layer_out_mem["feat.bmm1"],
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=64,
            K=3,
            a=layer_out_mem["feat.bmm1"],
            b=params["feat.conv1.weight"],
            c=layer_out_mem["feat.conv1"],
            v=params["feat.conv1.bias"],
            alpha=params["feat.bn1.weight"],
            beta=params["feat.bn1.bias"],
            activation="relu",
        )
        # feat.fstn
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=64,
            K=64,
            a=layer_out_mem["feat.conv1"],
            b=params["feat.fstn.conv1.weight"],
            c=layer_out_mem["feat.fstn.conv1"],
            v=params["feat.fstn.conv1.bias"],
            alpha=params["feat.fstn.bn1.weight"],
            beta=params["feat.fstn.bn1.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=128,
            K=64,
            a=layer_out_mem["feat.fstn.conv1"],
            b=params["feat.fstn.conv2.weight"],
            c=layer_out_mem["feat.fstn.conv2"],
            v=params["feat.fstn.conv2.bias"],
            alpha=params["feat.fstn.bn2.weight"],
            beta=params["feat.fstn.bn2.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=1024,
            K=128,
            a=layer_out_mem["feat.fstn.conv2"],
            b=params["feat.fstn.conv3.weight"],
            c=layer_out_mem["feat.fstn.conv3"],
            v=params["feat.fstn.conv3.bias"],
            alpha=params["feat.fstn.bn3.weight"],
            beta=params["feat.fstn.bn3.bias"],
            activation="relu",
        )
        max_dim1_caller(
            batch_size=batch_size,
            n=fixed_length,
            c=1024,
            x=layer_out_mem["feat.fstn.conv3"],
            out=layer_out_mem["feat.maxpool"],
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size,
            N=512,
            K=1024,
            a=layer_out_mem["feat.maxpool"],
            b=params["feat.fstn.fc1.weight"],
            c=layer_out_mem["feat.fstn.fc1"],
            v=params["feat.fstn.fc1.bias"],
            alpha=params["feat.fstn.bn4.weight"],
            beta=params["feat.fstn.bn4.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size,
            N=256,
            K=512,
            a=layer_out_mem["feat.fstn.fc1"],
            b=params["feat.fstn.fc2.weight"],
            c=layer_out_mem["feat.fstn.fc2"],
            v=params["feat.fstn.fc2.bias"],
            alpha=params["feat.fstn.bn5.weight"],
            beta=params["feat.fstn.bn5.bias"],
            activation="relu",
        )
        biased_gemm_fuse_addbasis_caller(
            M=batch_size,
            N=4096,
            K=256,
            r=64,
            a=layer_out_mem["feat.fstn.fc2"],
            b=params["feat.fstn.fc3.weight"],
            c=layer_out_mem["feat.fstn.fc3"],
            v=params["feat.fstn.fc3.bias"],
        )
        # feat
        batched_gemm_caller(
            batch_size=batch_size,
            M=fixed_length,
            N=64,
            K=64,
            a=layer_out_mem["feat.conv1"],
            b=layer_out_mem["feat.fstn.fc3"],
            c=layer_out_mem["feat.bmm2"],
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=128,
            K=64,
            a=layer_out_mem["feat.bmm2"],
            b=params["feat.conv2.weight"],
            c=layer_out_mem["feat.conv2"],
            v=params["feat.conv2.bias"],
            alpha=params["feat.bn2.weight"],
            beta=params["feat.bn2.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size * fixed_length,
            N=1024,
            K=128,
            a=layer_out_mem["feat.conv2"],
            b=params["feat.conv3.weight"],
            c=layer_out_mem["feat.conv3"],
            v=params["feat.conv3.bias"],
            alpha=params["feat.bn3.weight"],
            beta=params["feat.bn3.bias"],
        )
        max_dim1_caller(
            batch_size=batch_size,
            n=fixed_length,
            c=1024,
            x=layer_out_mem["feat.conv3"],
            out=layer_out_mem["feat.maxpool"],
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size,
            N=512,
            K=1024,
            a=layer_out_mem["feat.maxpool"],
            b=params["fc1.weight"],
            c=layer_out_mem["fc1"],
            v=params["fc1.bias"],
            alpha=params["bn1.weight"],
            beta=params["bn1.bias"],
            activation="relu",
        )
        biased_gemm_fuse_pbn_caller(
            M=batch_size,
            N=256,
            K=512,
            a=layer_out_mem["fc1"],
            b=params["fc2.weight"],
            c=layer_out_mem["fc2"],
            v=params["fc2.bias"],
            alpha=params["bn2.weight"],
            beta=params["bn2.bias"],
            activation="relu",
        )
        biased_gemm_caller(
            M=batch_size,
            N=10,
            K=256,
            a=layer_out_mem["fc2"],
            b=params["fc3.weight"],
            c=layer_out_mem["fc3"][idx : idx + batch_size],
            v=params["fc3.bias"],
        )
        idx += batch_size
        if idx + batch_size > dataset_size:
            batch_size = dataset_size - idx
    ## 3. 计算准确率
    # torch_out = F.log_softmax(layer_out_mem["fc3"], dim=1, dtype=dtype)
    # torch_out = torch_out.data.max(1)[1]
    # cmp = torch_out == list_of_labels
    # dump_np(layer_out_mem["fc3"].cpu().numpy(), "triton.txt")
    logsoftmax_dim1_caller(layer_out_mem["fc3"], logsoftmax_out)
    argmax_dim1_caller(logsoftmax_out, inference_labels)
    cmp = inference_labels == list_of_labels
    accuracy_rate = cmp.sum().item() / list_of_labels.shape[0]
    return accuracy_rate


if __name__ == "__main__":
    if DBG_FLAG:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在的目录
        dir = script_dir + "/../param/triton_param"
        dataPath = script_dir + "/../data/test_point_clouds.h5"
    else:
        dir = os.path.dirname(__file__)  # 保存模型参数文件(.txt)的文件夹路径
        dataPath = "./data/test_point_clouds.h5"

    # 推理参数
    batch_size = 1000
    fixed_length = 128
    num_class = 10

    # 读取模型参数
    params = read_params(dir)
    params: Dict[str, torch.Tensor] = preprocess_params(params=params, type=DTYPE)
    # 读取训练集数据
    list_of_points, list_of_labels = read_h5_file(dataPath, fixed_length)
    # 分配内存
    set_layer_out_shape(
        batch_size=batch_size, n=fixed_length, total_points=list_of_points.shape[0]
    )
    layer_out_mem = alloc_layer_out_tensor_mem(dtype=DTYPE)

    # to cuda
    list_of_labels = torch.tensor(list_of_labels, dtype=torch.int32).cuda()
    inference_labels = torch.empty_like(list_of_labels, dtype=torch.int32).cuda()
    logsoftmax_out = torch.empty(
        (list_of_points.shape[0], num_class), dtype=DTYPE
    ).cuda()
    # 开始计时
    start = time.time()
    accuracy_rate = do_inference(
        list_of_points=list_of_points,
        list_of_labels=list_of_labels,
        params=params,
        batch_size=batch_size,
        fixed_length=fixed_length,
        layer_out_mem=layer_out_mem,
        logsoftmax_out=logsoftmax_out,
        inference_labels=inference_labels,
        dtype=DTYPE,
    )
    # 结束计时
    end = time.time()
    ms = end - start
    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")
