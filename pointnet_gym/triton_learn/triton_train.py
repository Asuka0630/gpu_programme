# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习训练过程，请严格保持输出格式输出
from typing import Optional, Any, Union, List, Tuple, Dict
import os
import h5py
import time
import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

import triton
import triton.language as tl
from triton.runtime import driver
from triton import cdiv, next_power_of_2
from triton.language.extra import libdevice

Context = Any
Device = Optional[Union[torch.device, str]]
debug = True


### tools
def make_2d_for_mm(input: Tensor) -> Tensor:
    """
    Converts the input to a 2D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 1:
        input = input.unsqueeze(0)
    elif input.ndim >= 3:
        input = input.flatten(0, -2)
    return input


def make_3d_for_bn(input: Tensor) -> Tensor:
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim == 4:
        input = input.flatten(2, -1)

    return input


def make_3d_tensor_for_reduce(input: Tensor, dim: int) -> Tensor:
    """
    Make a nD tensor to a 3D tensor for the max reduce operation.

    Note: keep dim in the middle.
    """
    if dim < 0:
        dim = input.dim() + dim
    if dim >= input.dim():
        raise ValueError(f"dim={dim} should be less than input.dim() {input.dim()}")
    if input.dim() == 2:
        if dim == 0:
            return input.unsqueeze(0)
        return input.unsqueeze(-1)
    # Get the shape of the input tensor.
    if dim == 0:
        return make_3d_tensor_for_reduce(input.reshape(input.shape[0], -1), dim)
    elif dim == input.dim() - 1:
        return make_3d_tensor_for_reduce(input.reshape(-1, input.shape[-1]), 1)
    shape = (
        np.prod(input.shape[0:dim]),
        input.shape[dim],
        np.prod(input.shape[dim + 1 :]),
    )
    return input.reshape(shape)


def get_act_func(act_func_name: Optional[str] = None) -> str:
    """
    Returns the name of the function.
    """
    param = None
    if act_func_name == "":
        act_func_name = None
    if act_func_name is not None and "_" in act_func_name:
        comps = act_func_name.split("_")
        act_func_name = "_".join(comps[:-1])
        param = float(comps[-1])
    return act_func_name, param


def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def element_wise_kernel_configs(
    block_name: str = "BLOCK_SIZE",
) -> List[triton.Config]:
    return [
        triton.Config({block_name: 256}, num_warps=2),
        triton.Config({block_name: 512}, num_warps=4),
        triton.Config({block_name: 1024}, num_warps=4),
    ]


def matmul_kernel_configs() -> List[triton.Config]:
    # most frequently used configurations
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]


def warps_kernel_configs() -> List[triton.Config]:
    return [triton.Config({}, num_warps=2**i) for i in range(4)]


def reduce3d_kernel_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_SIZE_ROW": 128, "BLOCK_SIZE_COL": 128}, num_warps=4, num_stages=3
        ),
    ]


### kernels
@triton.jit
def relu_grad(input):
    return tl.where(input <= 0, 0, 1)


@triton.jit
def relu(input):
    return tl.maximum(0, input)


@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))


@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, output_grad / (1 - drop_p))


@triton.jit
def apply_act_func_grad(
    output_grad,
    input,
    drop_p,
    seed,
    offset,
    act_param,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
):
    if act_func == "relu":
        output = relu_grad(input)

    if dropout:
        output_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)

    return output_grad * output


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def act_func_backward_kernel(
    output_grad_pointer,
    input_pointer,
    input_grad_pointer,
    size,
    drop_p,
    seed,
    act_param,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)

    tl.store(
        input_grad_pointer + offset,
        apply_act_func_grad(
            output_grad, input, drop_p, seed, offset, act_param, act_func, dropout
        ),
        mask=mask,
    )


@triton.jit
def apply_act_func(
    input,
    drop_p,
    seed,
    offset,
    act_param,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
):
    if act_func == "relu":
        output = relu(input)

    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)

    return output


@triton.autotune(
    configs=matmul_kernel_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics({"tf32": lambda _: allow_tf32()})
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    v_ptr,
    pre_act_ptr,  # axb+v
    out_ptr,  # act_func(axb+v)
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_pre_act_m,
    stride_pre_act_n,
    stride_cm,
    stride_cn,
    # fuse option
    add_bias: tl.constexpr,
    act_param: float,
    act_func: tl.constexpr,
    save_pre_act: tl.constexpr,
    fp16: tl.constexpr,
    tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
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

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_m_masks = (offs_m < M)[:, None]
    offs_n_masks = (offs_n < N)[None, :]
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs,
            mask=offs_m_masks & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_n_masks & (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, allow_tf32=tf32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if fp16:
        accumulator = accumulator.to(tl.float16)

    if add_bias:
        v_ptrs = v_ptr + offs_n
        v = tl.load(v_ptrs, mask=offs_n < N, other=0.0)
        if fp16:
            v = v.to(tl.float16)
        accumulator += v

    # You can fuse arbitrary activation functions here
    if act_func is not None:
        if save_pre_act:
            pre_act_ptrs = pre_act_ptr + (
                offs_m[:, None] * stride_pre_act_m + offs_n[None, :] * stride_pre_act_n
            )
            tl.store(pre_act_ptrs, accumulator, mask=offs_m_masks & offs_n_masks)

        accumulator = apply_act_func(
            accumulator, None, None, None, act_param, act_func, False
        )

    out_ptrs = out_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    # Write back the block of the output matrix C with masks.
    tl.store(out_ptrs, accumulator, mask=offs_m_masks & offs_n_masks)


@triton.autotune(
    configs=matmul_kernel_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics({"tf32": lambda _: allow_tf32()})
@triton.jit
def batched_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    pre_act_ptr,
    c_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_a_batch,
    stride_am,
    stride_ak,
    stride_b_batch,
    stride_bk,
    stride_bn,
    stride_pre_act_batch,
    stride_pre_act_m,
    stride_pre_act_n,
    stride_c_batch,
    stride_cm,
    stride_cn,
    stride_bias_batch,
    stride_bias_feat,
    # precision
    bias_dim: tl.constexpr,
    fp16: tl.constexpr,
    tf32: tl.constexpr,
    act_param: tl.constexpr,
    act_func: tl.constexpr,  #
    save_pre_act: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
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

    a_batch_base_offset = batch_idx * stride_a_batch
    b_batch_base_offset = batch_idx * stride_b_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = (
        a_ptr
        + a_batch_base_offset
        + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        b_ptr
        + b_batch_base_offset
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_n[None, :] & (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # if fp16:
        #     a = a.to(tl.float16)
        #     b = b.to(tl.float16)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, allow_tf32=tf32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias_dim >= 0:
        bias_ptr += stride_bias_batch * batch_idx
        if bias_dim == 0:
            bias_ptrs = bias_ptr + offs_m[:, None] * stride_bias_feat
            bias = tl.load(bias_ptrs, mask=mask_m[:, None])
        else:
            bias_ptrs = bias_ptr + offs_n[None, :] * stride_bias_feat
            bias = tl.load(bias_ptrs, mask=mask_n[None, :])
        accumulator += bias

    ## You can fuse arbitrary activation functions here
    pre_act_batch_base_offset = batch_idx * stride_pre_act_batch
    if act_func is not None:
        if save_pre_act:
            pre_act_ptrs = (
                pre_act_ptr
                + pre_act_batch_base_offset
                + (
                    offs_m[:, None] * stride_pre_act_m
                    + offs_n[None, :] * stride_pre_act_n
                )
            )
            tl.store(
                pre_act_ptrs,
                accumulator,
                mask=mask_m[:, None] & mask_n[None, :],
            )

        accumulator = apply_act_func(
            accumulator, None, None, None, act_param, act_func, False
        )
    if fp16:
        accumulator = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_batch_base_offset = batch_idx * stride_c_batch
    c_ptrs = (
        c_ptr
        + c_batch_base_offset
        + (stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :])
    )
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


def BLOCK_SIZE_SPATIAL_heuristic(args: Dict) -> int:
    BLOCK_SIZE_BATCH = next_power_of_2(args["batch_dim"])
    BLOCK_SIZE_SPATIAL = next_power_of_2(args["spatial_dim"])
    return min(BLOCK_SIZE_SPATIAL, max(1, 2**14 // BLOCK_SIZE_BATCH))


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": lambda args: next_power_of_2(args["batch_dim"]),
        "BLOCK_SIZE_SPATIAL": BLOCK_SIZE_SPATIAL_heuristic,
    }
)
@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    pre_act_add_pointer,
    pre_act_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    pre_act_add_batch_stride,
    pre_act_add_feat_stride,
    pre_act_add_spatial_stride,
    pre_act_batch_stride,
    pre_act_feat_stride,
    pre_act_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    act_param,
    affine: tl.constexpr,
    save_stats: tl.constexpr,
    track_running_stats: tl.constexpr,
    is_train: tl.constexpr,
    add_pre_act: tl.constexpr,
    act_func: tl.constexpr,
    save_pre_act: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim

    if is_train or not track_running_stats:
        count = 0
        mean = 0.0
        var = 0.0

        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
                0, BLOCK_SIZE_SPATIAL
            )
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)

            spatial_count = min(
                BLOCK_SIZE_SPATIAL, spatial_dim - block_ind * BLOCK_SIZE_SPATIAL
            )
            curr_count = spatial_count * batch_dim
            count += curr_count

            prev_mean = mean
            mean += (tl.sum(curr_input) - curr_count * mean) / count
            deltas = tl.where(
                batch_mask[:, None] & spatial_mask[None, :],
                (curr_input - mean) * (curr_input - prev_mean),
                0.0,
            )
            var += tl.sum(deltas)

        var /= count
        inv_std = tl.rsqrt(var + eps)

        if save_stats:
            tl.store(feat_pid + mean_pointer, mean)
            tl.store(feat_pid + inv_std_pointer, inv_std)

        if track_running_stats:
            running_mean_pointer += feat_pid
            running_var_pointer += feat_pid

            running_mean = tl.load(running_mean_pointer)
            running_var = tl.load(running_var_pointer)

            n = batch_dim * spatial_dim
            tl.store(
                running_mean_pointer, (1 - momentum) * running_mean + momentum * mean
            )
            tl.store(
                running_var_pointer,
                (1 - momentum) * running_var + momentum * var * n / (n - 1),
            )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = tl.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        bias = tl.load(feat_pid + bias_pointer)

    else:
        weight = 1.0
        bias = 0.0

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
            0, BLOCK_SIZE_SPATIAL
        )
        spatial_mask = spatial_offset < spatial_dim

        curr_input_pointer = (
            input_pointer
            + input_feat_stride * feat_pid
            + input_batch_stride * batch_offset[:, None]
            + input_spatial_stride * spatial_offset[None, :]
        )
        curr_output_pointer = (
            output_pointer
            + output_feat_stride * feat_pid
            + output_batch_stride * batch_offset[:, None]
            + output_spatial_stride * spatial_offset[None, :]
        )

        curr_input = tl.load(
            curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        output = weight * (curr_input - mean) * inv_std + bias

        if add_pre_act:
            curr_pre_act_add_pointer = (
                pre_act_add_pointer
                + pre_act_add_feat_stride * feat_pid
                + pre_act_add_batch_stride * batch_offset[:, None]
                + pre_act_add_spatial_stride * spatial_offset[None, :]
            )
            curr_pre_act_add = tl.load(
                curr_pre_act_add_pointer,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )
            output += curr_pre_act_add

        if act_func is not None:
            if save_pre_act:
                curr_pre_act_pointer = (
                    pre_act_pointer
                    + pre_act_feat_stride * feat_pid
                    + pre_act_batch_stride * batch_offset[:, None]
                    + pre_act_spatial_stride * spatial_offset[None, :]
                )
                tl.store(
                    curr_pre_act_pointer,
                    output,
                    mask=batch_mask[:, None] & spatial_mask[None, :],
                )

            output = apply_act_func(
                output, None, None, None, act_param, act_func, False
            )

        tl.store(
            curr_output_pointer,
            output,
            mask=batch_mask[:, None] & spatial_mask[None, :],
        )


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": lambda args: next_power_of_2(args["batch_dim"]),
        "BLOCK_SIZE_SPATIAL": BLOCK_SIZE_SPATIAL_heuristic,
    }
)
@triton.jit
def batch_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    affine: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim

    mean = tl.load(feat_pid + mean_pointer)
    inv_std = tl.load(feat_pid + inv_std_pointer)

    term1 = 0.0  # \sum{x_hat * dy}
    term2 = 0.0  # \sum{dy}

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
            0, BLOCK_SIZE_SPATIAL
        )
        spatial_mask = spatial_offset < spatial_dim

        curr_output_grad_pointer = (
            output_grad_pointer
            + output_grad_feat_stride * feat_pid
            + output_grad_batch_stride * batch_offset[:, None]
            + output_grad_spatial_stride * spatial_offset[None, :]
        )
        curr_input_pointer = (
            input_pointer
            + input_feat_stride * feat_pid
            + input_batch_stride * batch_offset[:, None]
            + input_spatial_stride * spatial_offset[None, :]
        )

        curr_input = tl.load(
            curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(
            curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)

        term1 += tl.sum(curr_pre_lin * curr_output_grad)
        term2 += tl.sum(curr_output_grad)

    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        weight_grad = 0.0
        bias_grad = 0.0

    else:
        weight = 1.0

    count = batch_dim * spatial_dim
    term1 *= weight / count  # \bar{x_hat * (dy*w)} = \sum{x_hat * dy} * w/(B*N)
    term2 *= weight / count  # \bar{dy*w} = \sum{dy} * w/(B*N)

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
            0, BLOCK_SIZE_SPATIAL
        )
        spatial_mask = spatial_offset < spatial_dim

        curr_output_grad_pointer = (
            output_grad_pointer
            + output_grad_feat_stride * feat_pid
            + output_grad_batch_stride * batch_offset[:, None]
            + output_grad_spatial_stride * spatial_offset[None, :]
        )
        curr_input_pointer = (
            input_pointer
            + input_feat_stride * feat_pid
            + input_batch_stride * batch_offset[:, None]
            + input_spatial_stride * spatial_offset[None, :]
        )
        curr_input_grad_pointer = (
            input_grad_pointer
            + input_grad_feat_stride * feat_pid
            + input_grad_batch_stride * batch_offset[:, None]
            + input_grad_spatial_stride * spatial_offset[None, :]
        )

        curr_input = tl.load(
            curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(
            curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        curr_input_grad = inv_std * (
            weight * curr_output_grad - (term1 * curr_pre_lin + term2)
        )  # dx_hat = weight * curr_output_grad
        tl.store(
            curr_input_grad_pointer,
            curr_input_grad,
            mask=batch_mask[:, None] & spatial_mask[None, :],
        )

        if affine:
            weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
            bias_grad += tl.sum(curr_output_grad)

    if affine:
        tl.store(feat_pid + weight_grad_pointer, weight_grad)
        tl.store(feat_pid + bias_grad_pointer, bias_grad)


@triton.autotune(
    configs=reduce3d_kernel_configs(),
    key=["shape1", "shape2"],
)
@triton.jit
def max_kernels(
    input_ptr,
    output_ptr,
    indice_ptr,
    shape1,
    shape2,
    in_stride0,
    in_stride1,
    in_stride2,
    out_stride0,
    out_stride1,
    ind_stride0,
    ind_stride1,
    tracking_indices: tl.constexpr,
    fp16: tl.constexpr,
    ind_i64: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    # 一个pid负责reduce一个batch内的所有BLOCK_SIZE列
    pid = tl.program_id(0)
    # 计算当前pid对应的batch和列
    num_pid_col = tl.cdiv(shape2, BLOCK_SIZE_COL)
    batch_idx = pid // num_pid_col
    col_pid_idx = pid % num_pid_col
    # offset
    input_ptr += batch_idx * in_stride0
    col_offs = col_pid_idx * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_masks = col_offs < shape2
    accumulator = tl.full((BLOCK_SIZE_COL,), -float("inf"), dtype=tl.float32)
    if tracking_indices:
        accumulator_ind = tl.zeros(
            (BLOCK_SIZE_COL,), dtype=tl.int32 if not ind_i64 else tl.int64
        )
    for row_idx in range(0, shape1, BLOCK_SIZE_ROW):
        # load block
        row_offs = row_idx + tl.arange(0, BLOCK_SIZE_ROW)
        row_masks = row_offs < shape1
        input_ptrs = (
            input_ptr + row_offs[:, None] * in_stride1 + col_offs[None, :] * in_stride2
        )
        blk_max = tl.load(
            input_ptrs,
            mask=row_masks[:, None] & col_masks[None, :],
            other=-float("inf"),
        )
        blk_max, blk_max_ind = tl.max(blk_max, axis=0, return_indices=True)
        # update column-wise max of current block into global accumulator
        if tracking_indices:
            accumulator_ind = tl.where(
                blk_max > accumulator, blk_max_ind + row_idx, accumulator_ind
            )
        accumulator = tl.where(blk_max > accumulator, blk_max, accumulator)
    ## store
    output_ptrs = output_ptr + batch_idx * out_stride0 + col_offs * out_stride1
    if fp16:
        accumulator = accumulator.to(dtype=tl.float16)
    tl.store(
        output_ptrs,
        accumulator,
        mask=tl.arange(0, BLOCK_SIZE_COL) < shape2 - col_pid_idx * BLOCK_SIZE_COL,
    )
    if tracking_indices:
        indice_ptrs = indice_ptr + batch_idx * ind_stride0 + col_offs * ind_stride1
        tl.store(
            indice_ptrs,
            accumulator_ind,
            mask=tl.arange(0, BLOCK_SIZE_COL) < shape2 - col_pid_idx * BLOCK_SIZE_COL,
        )


@triton.autotune(
    # configs=reduce3d_kernel_configs(),
    # bugs for some specific num_warps
    # https://github.com/triton-lang/triton/issues/5327
    configs=[
        triton.Config(
            {"BLOCK_SIZE_ROW": 64, "BLOCK_SIZE_COL": 128}, num_warps=4, num_stages=3
        ),
    ],
    key=["shape1", "shape2"],
)
@triton.jit
def mean_kernels(
    input_ptr,
    output_ptr,
    shape1,
    shape2,
    in_stride0,
    in_stride1,
    in_stride2,
    out_stride0,
    out_stride1,
    avg: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    # 一个pid负责reduce一个batch内的所有BLOCK_SIZE列
    pid = tl.program_id(0)
    # 计算当前pid对应的batch和列
    num_pid_col = tl.cdiv(shape2, BLOCK_SIZE_COL)
    batch_idx = pid // num_pid_col
    col_pid_idx = pid % num_pid_col
    # offset
    input_ptr += batch_idx * in_stride0
    col_offs = col_pid_idx * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_masks = col_offs < shape2
    accumulator = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.float32)

    for row_idx in range(0, shape1, BLOCK_SIZE_ROW):
        # load bock
        row_offs = row_idx + tl.arange(0, BLOCK_SIZE_ROW)
        row_masks = row_offs < shape1
        input_ptrs = (
            input_ptr + row_offs[:, None] * in_stride1 + col_offs[None, :] * in_stride2
        )
        blk_sum = tl.load(
            input_ptrs,
            mask=row_masks[:, None] & col_masks[None, :],
            other=0.0,
        ).to(tl.float32)
        # update column-wise sum of current block into global accumulator
        accumulator = accumulator + tl.sum(blk_sum, axis=0)
    if avg:
        accumulator = accumulator / shape1
    ## store
    output_ptrs = output_ptr + batch_idx * out_stride0 + col_offs * out_stride1
    if fp16:
        accumulator = accumulator.to(dtype=tl.float16)
    tl.store(
        output_ptrs,
        accumulator,
        mask=tl.arange(0, BLOCK_SIZE_COL) < shape2 - col_pid_idx * BLOCK_SIZE_COL,
    )


@triton.autotune(configs=[triton.Config({}, num_stages=3, num_warps=4)], key=["nums"])
@triton.heuristics(
    {
        "BLOCK_SIZE": lambda args: next_power_of_2(args["nums"]),
    }
)
@triton.jit
def pnorm_forward_kernels(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    output_stride,
    nums,
    p: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr += pid * input_stride0
    n_offs = tl.arange(0, BLOCK_SIZE)
    mask = n_offs < nums
    input_ptrs = input_ptr + n_offs * input_stride1
    data = tl.load(input_ptrs, mask=mask, other=0.0)
    data = libdevice.pow(tl.abs(data) + 1e-5, p)
    accmu = tl.sum(data)
    accmu = libdevice.pow(accmu, 1.0 / p)
    if fp16:
        accmu = accmu.to(dtype=tl.float16)
    tl.store(output_ptr + pid * output_stride, accmu)


@triton.autotune(configs=[triton.Config({}, num_stages=3, num_warps=2)], key=["nums"])
@triton.heuristics(
    {
        "BLOCK_SIZE": lambda args: next_power_of_2(args["nums"]),
    }
)
@triton.jit
def pnorm_backward_kernels(
    input_grad_ptr,
    output_grad_ptr,
    output_ptr,
    input_ptr,
    input_grad_stride0,
    input_grad_stride1,
    output_grad_stride,
    output_stride,
    input_stride0,
    input_stride1,
    nums,
    p: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_grad_ptr += pid * input_grad_stride0
    output_grad_ptr += pid * output_grad_stride
    output_ptr += pid * output_stride
    input_ptr += pid * input_stride0
    n_offs = tl.arange(0, BLOCK_SIZE)
    mask = n_offs < nums
    input_ptrs = input_ptr + n_offs * input_stride1
    input_data = tl.load(input_ptrs, mask=mask, other=0.0)  # load x
    output_data = tl.load(output_ptr)  # load y
    output_grad_data = tl.load(output_grad_ptr)  # load dy
    # dy * y^{1-p} * x * |x|^{p-2}
    input_grad_data = (
        output_grad_data
        * libdevice.pow(output_data, (1 - p))
        * input_data
        * libdevice.pow(tl.abs(input_data), (p - 2))
    )
    if fp16:
        input_grad_data = input_grad_data.to(dtype=tl.float16)
    input_grad_ptrs = input_grad_ptr + n_offs * input_grad_stride1
    tl.store(input_grad_ptrs, input_grad_data, mask=mask)


@triton.jit
def softmax_dim1_forward_kernel(
    input_ptr,
    output_ptr,
    n_rows: int,
    n_cols: int,
    input_stride_row: int,
    input_stride_col: int,
    output_stride_row: int,
    output_stride_col: int,
    log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0).to(tl.int64)
    row_step = tl.num_programs(0).to(tl.int64)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_stride_row
        """
        BLOCK_SIZE是最小的大于等于n_cols的2的幂
        """
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets * input_stride_col
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
        # softmax_out = exp(x-x.max())/sum(exp(x-x.max())) = exp(row_minus_max)/sum(exp(row_minus_max))
        row_minus_max = row - tl.max(row, axis=0)
        denominator = tl.sum(tl.exp(row_minus_max), axis=0)
        if log:
            # logsoftmax_out = (x-x.max())/ln(sum(exp(x-x.max()))) = row_minus_max/ln(sum(exp(row_minus_max)))
            output = row_minus_max - tl.log(denominator + 1e-8)
        else:
            output = tl.exp(row_minus_max) / (denominator + 1e-8)

        output_row_start_ptr = output_ptr + row_idx * output_stride_row
        output_ptrs = output_row_start_ptr + col_offsets * output_stride_col
        tl.store(output_ptrs, output, mask=col_offsets < n_cols)


@triton.jit
def softmax_dim1_backward_kernel(
    output_ptr,
    output_grad_ptr,
    x_grad_ptr,
    n_rows: int,
    n_cols: int,
    output_stride_row: int,
    output_stride_col: int,
    grad_output_stride_row: int,
    grad_output_stride_col: int,
    grad_x_stride_row: int,
    grad_x_stride_col: int,
    log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0).to(tl.int64)
    row_step = tl.num_programs(0).to(tl.int64)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        """
        BLOCK_SIZE是最小的大于等于n_cols的2的幂
        """
        col_offsets = tl.arange(0, BLOCK_SIZE)
        output_ptrs = (
            output_ptr + row_idx * output_stride_row + col_offsets * output_stride_col
        )
        y = tl.load(output_ptrs, mask=col_offsets < n_cols)
        output_grad_ptrs = (
            output_grad_ptr
            + row_idx * grad_output_stride_row
            + col_offsets * grad_output_stride_col
        )
        dy = tl.load(output_grad_ptrs, mask=col_offsets < n_cols)
        if log:
            x_grad = dy - tl.exp(y) * tl.sum(dy)
        else:
            x_grad = y * (dy - tl.sum(dy * y))
        x_grad_ptrs = (
            x_grad_ptr + row_idx * grad_x_stride_row + col_offsets * grad_x_stride_col
        )
        tl.store(x_grad_ptrs, x_grad, mask=col_offsets < n_cols)


softmax_warm_kernels = {}


def softmax_dim1_warmup(x: torch.Tensor, log: bool, d: str = ""):
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)  # 获取设备参数
    NUM_SM = properties["multiprocessor_count"]  # SM数量
    NUM_REGS = properties["max_num_regs"]  # 每个SM上的寄存器数量
    SIZE_SMEM = properties["max_shared_mem"]  # 每个SM上的共享内存大小
    WARP_SIZE = properties["warpSize"]  # 每个warp的线程数
    n_rows, n_cols = x.shape
    global softmax_warm_kernels
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    ky = str(BLOCK_SIZE) + str(log)
    if (ky + "bwd") in softmax_warm_kernels and (ky + "fwd") in softmax_warm_kernels:
        return
    # Allocate output
    y = torch.empty_like(x)
    num_warps = 8
    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    if (ky + "fwd") not in softmax_warm_kernels:
        # pre-compile forward kernel to get register usage and compute thread occupancy.
        fwd_kernel = softmax_dim1_forward_kernel.warmup(
            input_ptr=x,
            output_ptr=y,
            n_rows=n_rows,
            n_cols=n_cols,
            input_stride_row=x.stride(0),
            input_stride_col=x.stride(1),
            output_stride_row=y.stride(0),
            output_stride_col=y.stride(1),
            log=log,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        fwd_kernel._init_handles()
        """
        kernel.n_regs应该是一个线程需要的寄存器数
        """
        n_regs = max(1, fwd_kernel.n_regs)
        size_smem = max(1, fwd_kernel.metadata.shared)
        """
        这里occypancy表示一个处理器上能并行的block数
        """
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        softmax_warm_kernels[ky + "fwd"] = (fwd_kernel, num_programs)
        if d == "fwd":
            return fwd_kernel, num_programs
    if (ky + "bwd") not in softmax_warm_kernels:
        dy = torch.empty_like(y)
        dx = y
        # pre-compile backward kernel to get register usage and compute thread occupancy.
        bwd_kernel = softmax_dim1_backward_kernel.warmup(
            output_ptr=y,
            output_grad_ptr=dy,
            x_grad_ptr=dx,
            n_rows=n_rows,
            n_cols=n_cols,
            output_stride_row=y.stride(0),
            output_stride_col=y.stride(1),
            grad_output_stride_row=dy.stride(0),
            grad_output_stride_col=dy.stride(1),
            grad_x_stride_row=dx.stride(0),
            grad_x_stride_col=dx.stride(1),
            log=log,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        bwd_kernel._init_handles()
        """
        kernel.n_regs应该是一个线程需要的寄存器数
        """
        n_regs = max(1, bwd_kernel.n_regs)
        size_smem = max(1, bwd_kernel.metadata.shared)
        """
        这里occypancy表示一个处理器上能并行的block数
        """
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        softmax_warm_kernels[ky + "bwd"] = (bwd_kernel, num_programs)
        if d == "bwd":
            return bwd_kernel, num_programs


def BLOCK_SIZE_BATCH_heuristic(args) -> int:
    """
    Approximates an appropriate batch block size for NLL loss using a heuristic.

    Args:
        args: Arguments to NLL loss kernel.

    Returns:
        Appropriate batch block size.
    """
    block_size_batch = max(1, triton.next_power_of_2(args["batch_dim"] // 2**10))
    block_size_batch = min(block_size_batch, 128)
    return block_size_batch if args["spatial_dim"] < 64 else 1


@triton.autotune(configs=warps_kernel_configs(), key=["batch_dim", "spatial_dim"])
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_SPATIAL": lambda args: triton.next_power_of_2(args["spatial_dim"]),
    }
)
@triton.jit
def nll_loss_forward_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    sum_weight_ptr,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    target_batch_stride,
    target_spatial_stride,
    weight_stride,
    output_batch_stride,
    output_spatial_stride,
    sum_weight_stride,
    fp16: tl.constexpr,
    reduction: tl.constexpr,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    # one program processes BLOCK_SIZE_BATCH batches and BLOCK_SIZE_SPATIAL(>=spatial_dim) elements
    batch_pid = tl.program_id(0).to(tl.int64)
    batch_offs = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH).to(
        tl.int64
    )

    batch_masks = batch_offs < batch_dim

    spatial_offs = tl.arange(0, BLOCK_SIZE_SPATIAL).to(
        tl.int64
    )  # next_power_of_2(spatial_dim)

    spatial_masks = spatial_offs < spatial_dim

    target_ptrs = (
        target_ptr
        + batch_offs[:, None] * target_batch_stride
        + spatial_offs[None, :] * target_spatial_stride
    )
    target = tl.load(
        target_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
    ).to(tl.int64)

    """
    在一个Batch内, input_ptr的排布是每行表示一个特征维度, 每列表示一个样本
    """
    input_ptrs = (
        input_ptr
        + batch_offs[:, None] * input_batch_stride
        + spatial_offs[None, :] * input_spatial_stride
        + target * input_feat_stride  # this actually do gather operation
    )

    input = tl.load(input_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]).to(
        tl.float32
    )

    output = -input  # here is the unreducted negative log likelihood loss
    if weighted:
        weight_ptrs = weight_ptr + target * weight_stride
        weight = tl.load(
            weight_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
        ).to(tl.float32)
        output *= weight
    if reduction == "none":
        output_ptrs = (
            output_ptr
            + batch_offs[:, None] * output_batch_stride
            + spatial_offs[None, :] * output_spatial_stride
        )
        if fp16:
            output = output.to(tl.float16)
        tl.store(
            output_ptrs, output, mask=batch_masks[:, None] & spatial_masks[None, :]
        )
    else:
        output_sum = tl.sum(output)
        if reduction == "mean":
            if not weighted:
                # weighted mean do not need div total seperately
                output_sum = output_sum / (batch_dim * spatial_dim)
            output_ptrs = output_ptr + batch_pid * output_batch_stride
            if fp16:
                output_sum = output_sum.to(tl.float16)
            tl.store(output_ptrs, output_sum)
            if weighted:
                # partly reduce
                sum_weight_ptrs = sum_weight_ptr + batch_pid * sum_weight_stride
                # store sum of weight to avoid sample nums affect weighted mean
                # e.g. for 2-classes weighted like[5, 1]
                # if in a batch there are 1 sample in class 0 and 5 samples in class 1
                # sum of weight = 5*1+1*5 = 10
                # so for each class, the weight is 5*1/10=0.5 and 1*5/10=0.5
                sum_weight = tl.sum(weight)
                if fp16:
                    sum_weight = sum_weight.to(tl.float16)
                tl.store(sum_weight_ptrs, sum_weight)
        elif reduction == "sum":
            output_ptrs = output_ptr + batch_pid * output_batch_stride  # partly reduce
            if fp16:
                output_sum = output_sum.to(tl.float16)
            tl.store(output_ptrs, output_sum)


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_SPATIAL": lambda args: triton.next_power_of_2(args["spatial_dim"]),
    }
)
@triton.jit
def nll_loss_backward_kernel(
    output_grad_ptr,
    target_ptr,
    weight_ptr,
    sum_weight_ptr,
    input_grad_ptr,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_spatial_stride,
    target_batch_stride,
    target_spatial_stride,
    weight_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    fp16: tl.constexpr,
    reduction: tl.constexpr,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    batch_offs = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    batch_masks = batch_offs < batch_dim

    spatial_offs = tl.arange(0, BLOCK_SIZE_SPATIAL)
    spatial_masks = spatial_offs < spatial_dim

    output_grad_masks = None
    output_grad_ptrs = output_grad_ptr
    if reduction == "none":
        output_grad_ptrs = (
            output_grad_ptr
            + batch_offs[:, None] * output_grad_batch_stride
            + spatial_offs[None, :] * output_grad_spatial_stride
        )
        output_grad_masks = batch_masks[:, None] & spatial_masks[None, :]

    output_grad = tl.load(output_grad_ptrs, mask=output_grad_masks).to(tl.float32)
    input_grad = -output_grad  # here we get naive nll loss of input_grad

    target_ptrs = (
        target_ptr
        + batch_offs[:, None] * target_batch_stride
        + spatial_offs[None, :] * target_spatial_stride
    )
    target = tl.load(
        target_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
    ).to(tl.int64)

    if weighted:
        weight_ptrs = weight_ptr + target * weight_stride  # gather operation
        weight = tl.load(
            weight_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
        ).to(tl.float32)
        input_grad *= weight
        if reduction == "mean":
            input_grad /= tl.load(sum_weight_ptr).to(tl.float32)
    elif reduction == "mean":
        input_grad /= batch_dim * spatial_dim

    input_grad_ptrs = (
        input_grad_ptr
        + batch_offs[:, None] * input_grad_batch_stride
        + spatial_offs[None, :] * input_grad_spatial_stride
        + target * input_grad_feat_stride
    )  # here actually do scatter operation
    if fp16:
        input_grad = input_grad.to(tl.float16)
    tl.store(
        input_grad_ptrs, input_grad, mask=batch_masks[:, None] & spatial_masks[None, :]
    )  # scatter operation


### autograd
class MaxReduceAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        keepdim: bool = False,
    ) -> Tensor:
        """
        Forward pass of the max reduce operation.
        """
        input_3d = make_3d_tensor_for_reduce(input, dim)
        shape_3d = input_3d.shape
        requires_grad = input.requires_grad
        out_2d = torch.empty(
            (shape_3d[0], shape_3d[2]), dtype=input.dtype, device=input.device
        )
        indices_2d = (
            torch.empty_like(out_2d, dtype=torch.int64, device=input.device)
            if requires_grad
            else None
        )
        grid = lambda META: (shape_3d[0] * cdiv(shape_3d[2], META["BLOCK_SIZE_COL"]),)
        max_kernels[grid](
            input_ptr=input_3d,
            output_ptr=out_2d,
            indice_ptr=indices_2d if requires_grad else None,
            shape1=shape_3d[1],
            shape2=shape_3d[2],
            in_stride0=input_3d.stride(0),
            in_stride1=input_3d.stride(1),
            in_stride2=input_3d.stride(2),
            out_stride0=out_2d.stride(0),
            out_stride1=out_2d.stride(1),
            ind_stride0=indices_2d.stride(0) if requires_grad else 0,
            ind_stride1=indices_2d.stride(1) if requires_grad else 0,
            tracking_indices=requires_grad,
            fp16=input.dtype is torch.float16,
            ind_i64=True,
        )
        out_shape = list(input.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(input.shape)
        out_2d = out_2d.view(out_shape)
        if requires_grad:
            indices_2d = indices_2d.view(out_shape)
            ctx.save_for_backward(indices_2d)
        return (out_2d, indices_2d)

    @staticmethod
    def backward(
        ctx: Context, output_grad: Tensor, indices_grad: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of the max reduce operation.

        args include indices_grad
        """
        (indices_2d,) = ctx.saved_tensors
        if not ctx.keepdim:
            indices_2d: torch.Tensor = indices_2d.unsqueeze(ctx.dim)
            output_grad = output_grad.unsqueeze(ctx.dim)

        grad_input = torch.zeros(
            ctx.input_shape, dtype=output_grad.dtype, device=output_grad.device
        )
        with torch.no_grad():
            grad_input.scatter_(ctx.dim, indices_2d, output_grad)
        # make grad_input to the same shape as the input, except the dim
        return grad_input, None, None


class MeanReduceAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        keepdim: bool = False,
    ) -> Tensor:
        input_3d = make_3d_tensor_for_reduce(input, dim)
        shape_3d = input_3d.shape
        out_2d = torch.zeros(
            (shape_3d[0], shape_3d[2]), dtype=input.dtype, device=input.device
        )
        grid = lambda META: (shape_3d[0] * cdiv(shape_3d[2], META["BLOCK_SIZE_COL"]),)
        mean_kernels[grid](
            input_ptr=input_3d,
            output_ptr=out_2d,
            shape1=shape_3d[1],
            shape2=shape_3d[2],
            in_stride0=input_3d.stride(0),
            in_stride1=input_3d.stride(1),
            in_stride2=input_3d.stride(2),
            out_stride0=out_2d.stride(0),
            out_stride1=out_2d.stride(1),
            avg=True,
            fp16=input.dtype is torch.float16,
        )
        out_shape = list(input.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(input.shape)
        out_2d = out_2d.view(out_shape)
        return out_2d

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not ctx.keepdim:
            output_grad = output_grad.unsqueeze(ctx.dim)

        grad_input = output_grad.expand(ctx.input_shape) / ctx.input_shape[ctx.dim]
        # make grad_input to the same shape as the input, except the dim
        return grad_input, None, None


class SumReduceAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        keepdim: bool = False,
    ) -> Tensor:
        input_3d = make_3d_tensor_for_reduce(input, dim)
        shape_3d = input_3d.shape
        out_2d = torch.zeros(
            (shape_3d[0], shape_3d[2]), dtype=input.dtype, device=input.device
        )
        grid = lambda META: (shape_3d[0] * cdiv(shape_3d[2], META["BLOCK_SIZE_COL"]),)
        mean_kernels[grid](
            input_ptr=input_3d,
            output_ptr=out_2d,
            shape1=shape_3d[1],
            shape2=shape_3d[2],
            in_stride0=input_3d.stride(0),
            in_stride1=input_3d.stride(1),
            in_stride2=input_3d.stride(2),
            out_stride0=out_2d.stride(0),
            out_stride1=out_2d.stride(1),
            avg=False,
            fp16=input.dtype is torch.float16,
        )
        out_shape = list(input.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(input.shape)
        out_2d = out_2d.view(out_shape)
        return out_2d

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not ctx.keepdim:
            output_grad = output_grad.unsqueeze(ctx.dim)

        grad_input = output_grad.expand(ctx.input_shape)
        # make grad_input to the same shape as the input, except the dim
        return grad_input, None, None


class NormReduceAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        p: float | str | None = 2.0,
        dim: Any | None = -1,
        keepdim: bool = False,
        out: Any | None = None,
        dtype: Any | None = torch.float32,
    ):
        if type(p) is str:
            raise ValueError(
                f"only `p-norm` is supported now, p should be a float, but got {type(p)}"
            )
        elif p == None:
            p = 2.0
        p = float(p)
        if dim == -1:
            dim = input.ndim - 1
        assert input.ndim - 1 == dim and "only support last dim norm now"
        out_shape = list(input.shape) if keepdim else list(input.shape)
        input_2d = input.flatten(0, -2)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        output = (
            out
            if out is not None
            else torch.empty(input_2d.shape[0], dtype=dtype, device=input.device)
        )
        requires_grad = input.requires_grad
        pnorm_forward_kernels[(input_2d.shape[0],)](
            input_ptr=input_2d,
            output_ptr=output,
            input_stride0=input_2d.stride(0),
            input_stride1=input_2d.stride(1),
            output_stride=output.stride(0),
            nums=input_2d.shape[1],
            p=p,
            fp16=dtype is torch.float16,
        )
        ctx.p = p
        if requires_grad:
            ctx.save_for_backward(input, output)
        return output.reshape(out_shape)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        input, output = ctx.saved_tensors
        input_2d: Tensor = input.flatten(0, -2)
        input_grad_2d = torch.empty_like(
            input_2d, dtype=output_grad.dtype, device=output_grad.device
        )
        output_grad_2d = output_grad.flatten()
        p = ctx.p
        pnorm_backward_kernels[(input_2d.shape[0],)](
            input_grad_ptr=input_grad_2d,
            output_grad_ptr=output_grad_2d,
            output_ptr=output,
            input_ptr=input_2d,
            input_grad_stride0=input_grad_2d.stride(0),
            input_grad_stride1=input_grad_2d.stride(1),
            output_grad_stride=output_grad_2d.stride(0),
            output_stride=output.stride(0),
            input_stride0=input_2d.stride(0),
            input_stride1=input_2d.stride(1),
            nums=input_2d.shape[1],
            p=p,
            fp16=output_grad.dtype is torch.float16,
        )
        return input_grad_2d.reshape(input.shape), None, None, None, None, None


class LinearAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        act_func: Optional[str] = None,
    ) -> Tensor:
        act_func, act_param = get_act_func(act_func)
        assert weight.ndim == 2, f"Weights must be 2D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"

        input_2d = make_2d_for_mm(input)

        assert (
            input_2d.shape[-1] == weight.shape[1]
        ), f"Incompatible input ({input_2d.shape}) and weights ({weight.shape}) shape"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), f"Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        M, K = input_2d.shape
        N, _ = weight.shape

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        save_pre_act = requires_grad and (act_func is not None)
        output_type = input.dtype
        output = torch.empty((M, N), device=input.device, dtype=output_type)
        pre_act = torch.empty_like(output) if save_pre_act else output

        grid = lambda META: (
            cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]),
        )
        matmul_kernel[grid](
            # Pointers to matrices
            a_ptr=input_2d,
            b_ptr=weight,
            v_ptr=bias,
            pre_act_ptr=pre_act,  # axb+v
            out_ptr=output,  # act_func(axb+v)
            # Matrix dimensions
            M=M,
            N=N,
            K=K,
            stride_am=input_2d.stride(0),
            stride_ak=input_2d.stride(1),
            stride_bk=weight.stride(1),
            stride_bn=weight.stride(0),
            stride_pre_act_m=pre_act.stride(0),
            stride_pre_act_n=pre_act.stride(1),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            # fuse option
            add_bias=bias is not None,
            act_param=act_param,
            act_func=act_func,
            save_pre_act=save_pre_act,
            fp16=output_type is torch.float16,
        )

        ctx.act_param = act_param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_type = output_type
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(input, pre_act if save_pre_act else None, weight)

        return output.view(*input.shape[:-1], N)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        input, pre_act, weight = ctx.saved_tensors

        output_grad_2d = make_2d_for_mm(output_grad)  # [M,N]
        input_2d = make_2d_for_mm(input)

        M, K = input_2d.shape
        N, _ = weight.shape

        assert (
            output_grad_2d.shape[0] == input_2d.shape[0]
            and output_grad_2d.shape[1] == weight.shape[0]
        ), f"Incompatible output gradient ({output_grad_2d.shape}), input ({input_2d.shape}) shape and weights ({weight.shape}) shape"

        if ctx.act_func is None:
            pre_act_grad = output_grad_2d
        else:
            size = M * N
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad,
                pre_act.view_as(pre_act_grad),
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )
            pre_act_grad = pre_act_grad.view_as(output_grad_2d)
        # dL/db = dL/dy
        bias_grad = _sum(pre_act_grad, dim=0) if ctx.bias_requires_grad else None
        if input.requires_grad:
            # dL/dx = dL/dy x W
            input_grad_2d = torch.empty((M, K), dtype=input.dtype, device=input.device)
            grid = lambda META: (
                cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(K, META["BLOCK_SIZE_N"]),
            )
            matmul_kernel[grid](
                pre_act_grad,
                weight,
                None,
                None,
                input_grad_2d,  # dL/dx
                M,
                K,
                N,
                stride_am=pre_act_grad.stride(0),
                stride_ak=pre_act_grad.stride(1),
                stride_bk=weight.stride(0),
                stride_bn=weight.stride(1),
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_cm=input_grad_2d.stride(0),
                stride_cn=input_grad_2d.stride(1),
                add_bias=False,
                act_param=None,
                act_func=None,
                save_pre_act=False,
                fp16=ctx.output_type is torch.float16,
            )
        else:
            input_grad_2d = None
        # dL/dW =dL/dy^T x X
        if weight.requires_grad:
            weight_grad = torch.empty_like(weight)
            grid = lambda META: (
                cdiv(N, META["BLOCK_SIZE_M"]) * cdiv(K, META["BLOCK_SIZE_N"]),
            )
            matmul_kernel[grid](
                pre_act_grad,
                input,
                None,
                None,
                weight_grad,  # dL/dW
                N,
                K,
                M,
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(0),
                stride_bk=input.stride(0),
                stride_bn=input.stride(1),
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_cm=weight_grad.stride(0),
                stride_cn=weight_grad.stride(1),
                add_bias=False,
                act_param=None,
                act_func=None,
                save_pre_act=False,
                fp16=ctx.output_type is torch.float16,
            )
        else:
            weight_grad = None
        bias_grad = _sum(pre_act_grad, dim=0) if ctx.bias_requires_grad else None
        return (
            input_grad_2d.view_as(input) if input_grad_2d is not None else None,
            weight_grad,
            bias_grad,
            None,
        )


class BatchMatmulAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: Tensor,
        b: Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False,
        # transpose_c: bool = False,
        act_func: Optional[str] = None,
    ) -> Tensor:
        B, M, Ka = a.shape
        if transpose_a:
            Ka, M = M, Ka
            transpose_c = True
        _, Kb, N = b.shape
        if transpose_b:
            N, Kb = Kb, N
        assert (
            Ka == Kb
            and a.shape[0] == b.shape[0]
            and f"Incompatible input ({a.shape}) and weights ({b.shape}) shape"
        )

        # if transpose_c:
        #     M, N = N, M

        grid = lambda META: (
            cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]),
            B,
        )
        act_func, act_param = get_act_func(act_func)
        save_pre_act = act_func is not None
        pre_act = (
            torch.empty((B, M, N), device=a.device, dtype=a.dtype)
            if save_pre_act
            else None
        )
        c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
        batched_gemm_kernel[grid](
            # Pointers to matrices
            a_ptr=a,
            b_ptr=b,
            pre_act_ptr=pre_act,
            c_ptr=c,
            bias_ptr=None,
            # Matrix dimensions
            M=M,
            N=N,
            K=Ka,
            stride_a_batch=a.stride(0),
            stride_am=a.stride(1) if not transpose_a else a.stride(2),
            stride_ak=a.stride(2) if not transpose_a else a.stride(1),
            stride_b_batch=b.stride(0),
            stride_bk=b.stride(1) if not transpose_b else b.stride(2),
            stride_bn=b.stride(2) if not transpose_b else b.stride(1),
            stride_pre_act_batch=pre_act.stride(0) if save_pre_act else 0,
            stride_pre_act_m=pre_act.stride(1) if pre_act is not None else 0,
            stride_pre_act_n=pre_act.stride(2) if pre_act is not None else 0,
            stride_c_batch=c.stride(0),
            stride_cm=c.stride(1),
            stride_cn=c.stride(2),
            stride_bias_batch=0,
            stride_bias_feat=0,
            # precision
            bias_dim=-1,
            fp16=a.dtype is torch.float16,
            act_param=act_param,
            act_func=act_func,
            save_pre_act=save_pre_act,
        )
        ctx.act_param = act_param
        ctx.act_func = act_func
        ctx.output_type = a.dtype
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(a, b, pre_act if save_pre_act else None)
        return c

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        B, M, N = output_grad.shape
        a, b, pre_act = ctx.saved_tensors
        if ctx.act_func is None:
            pre_act_grad = output_grad
        else:
            size = B * M * N
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad.flatten(),
                pre_act.view_as(pre_act_grad),
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )
            pre_act_grad = pre_act_grad.view_as(output_grad)

        grad_a = torch.empty_like(a, device=a.device, dtype=a.dtype)
        grad_b = torch.empty_like(b, device=b.device, dtype=b.dtype)

        grid_grad_a = lambda META: (
            cdiv(a.shape[1], META["BLOCK_SIZE_M"])
            * cdiv(a.shape[2], META["BLOCK_SIZE_N"]),
            B,
        )
        grid_grad_b = lambda META: (
            cdiv(b.shape[1], META["BLOCK_SIZE_M"])
            * cdiv(b.shape[2], META["BLOCK_SIZE_N"]),
            B,
        )

        if ctx.transpose_a:
            if ctx.transpose_b:
                """
                y = a^T · b^T
                dL/da = b^T·(dL/dy)^T
                dL/db = (dL/dy)^T·a^T
                """
                batched_gemm_kernel[grid_grad_a](
                    # Pointers to matrices
                    a_ptr=b,
                    b_ptr=pre_act_grad,
                    pre_act_ptr=None,
                    c_ptr=grad_a,  # [B,K,M]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=a.shape[1],  # K
                    N=a.shape[2],  # M
                    K=b.shape[1],  # N
                    stride_a_batch=b.stride(0),
                    stride_am=b.stride(2),
                    stride_ak=b.stride(1),
                    stride_b_batch=pre_act_grad.stride(0),
                    stride_bk=pre_act_grad.stride(2),
                    stride_bn=pre_act_grad.stride(1),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_a.stride(0),
                    stride_cm=grad_a.stride(1),
                    stride_cn=grad_a.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
                batched_gemm_kernel[grid_grad_b](
                    # Pointers to matrices
                    a_ptr=pre_act_grad,
                    b_ptr=a,
                    pre_act_ptr=None,
                    c_ptr=grad_b,  # [B,N,K]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=b.shape[1],  # N
                    N=b.shape[2],  # K
                    K=a.shape[2],  # M
                    stride_a_batch=pre_act_grad.stride(0),
                    stride_am=pre_act_grad.stride(2),
                    stride_ak=pre_act_grad.stride(1),
                    stride_b_batch=a.stride(0),
                    stride_bk=a.stride(2),
                    stride_bn=a.stride(1),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_b.stride(0),
                    stride_cm=grad_b.stride(1),
                    stride_cn=grad_b.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
            else:
                """
                y = a^T · b
                dL/da = b·(dL/dy)^T
                dL/db = a·dL/dy
                """
                batched_gemm_kernel[grid_grad_a](
                    # Pointers to matrices
                    a_ptr=b,
                    b_ptr=pre_act_grad,
                    pre_act_ptr=None,
                    c_ptr=grad_a,  # [B,K,M]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=a.shape[1],  # K
                    N=a.shape[2],  # M
                    K=b.shape[2],  # N
                    stride_a_batch=b.stride(0),
                    stride_am=b.stride(1),
                    stride_ak=b.stride(2),
                    stride_b_batch=pre_act_grad.stride(0),
                    stride_bk=pre_act_grad.stride(2),
                    stride_bn=pre_act_grad.stride(1),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_a.stride(0),
                    stride_cm=grad_a.stride(1),
                    stride_cn=grad_a.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
                batched_gemm_kernel[grid_grad_b](
                    # Pointers to matrices
                    a_ptr=a,
                    b_ptr=pre_act_grad,
                    pre_act_ptr=None,
                    c_ptr=grad_b,  # [B,K,N]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=b.shape[1],  # K
                    N=b.shape[2],  # N
                    K=a.shape[2],  # M
                    stride_a_batch=a.stride(0),
                    stride_am=a.stride(1),
                    stride_ak=a.stride(2),
                    stride_b_batch=pre_act_grad.stride(0),
                    stride_bk=pre_act_grad.stride(1),
                    stride_bn=pre_act_grad.stride(2),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_b.stride(0),
                    stride_cm=grad_b.stride(1),
                    stride_cn=grad_b.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
        elif ctx.transpose_b:
            """
            y = a · b^T
            dL/da = (dL/dy)·b
            dL/db = (dL/dy)^T·a
            """
            batched_gemm_kernel[grid_grad_a](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=b,
                pre_act_ptr=None,
                c_ptr=grad_a,  # [B,M,K]
                bias_ptr=None,
                # Matrix dimensions
                M=a.shape[1],  # M
                N=a.shape[2],  # K
                K=b.shape[1],  # N
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(2),
                stride_b_batch=b.stride(0),
                stride_bk=b.stride(1),
                stride_bn=b.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_a.stride(0),
                stride_cm=grad_a.stride(1),
                stride_cn=grad_a.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
            batched_gemm_kernel[grid_grad_b](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=a,
                pre_act_ptr=None,
                c_ptr=grad_b,  # [B,N,K]
                bias_ptr=None,
                # Matrix dimensions
                M=b.shape[1],  # N
                N=b.shape[2],  # K
                K=a.shape[1],  # M
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(2),
                stride_ak=pre_act_grad.stride(1),
                stride_b_batch=a.stride(0),
                stride_bk=a.stride(1),
                stride_bn=a.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_b.stride(0),
                stride_cm=grad_b.stride(1),
                stride_cn=grad_b.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
        else:
            """
            y = a · b
            dL/da = (dL/dy)·b^T
            dL/db = a^T·(dL/dy)
            """
            batched_gemm_kernel[grid_grad_a](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=b,
                pre_act_ptr=None,
                c_ptr=grad_a,  # [B,M,K]
                bias_ptr=None,
                # Matrix dimensions
                M=a.shape[1],  # M
                N=a.shape[2],  # K
                K=b.shape[2],  # N
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(2),
                stride_b_batch=b.stride(0),
                stride_bk=b.stride(2),
                stride_bn=b.stride(1),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_a.stride(0),
                stride_cm=grad_a.stride(1),
                stride_cn=grad_a.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
            batched_gemm_kernel[grid_grad_b](
                # Pointers to matrices
                a_ptr=a,
                b_ptr=pre_act_grad,
                pre_act_ptr=None,
                c_ptr=grad_b,  # [B,K,N]
                bias_ptr=None,
                # Matrix dimensions
                M=b.shape[1],  # K
                N=b.shape[2],  # N
                K=a.shape[1],  # M
                stride_a_batch=a.stride(0),
                stride_am=a.stride(2),
                stride_ak=a.stride(1),
                stride_b_batch=pre_act_grad.stride(0),
                stride_bk=pre_act_grad.stride(1),
                stride_bn=pre_act_grad.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_b.stride(0),
                stride_cm=grad_b.stride(1),
                stride_cn=grad_b.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
        return grad_a, grad_b, None, None, None, None


class Conv1d1kAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        act_func: Optional[str] = None,
    ) -> Tensor:
        act_func, act_param = get_act_func(act_func)

        assert input.ndim == 3, f"Input must be 3D, received shape {input.shape}"
        assert weight.ndim == 2, f"Weights must be 2D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), f"Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        out_feat, _ = weight.shape
        batch_size, in_feat, spatial_dim = input.shape

        assert (
            weight.shape[1] == in_feat
        ), f"Incompatible input ({input.shape}) and weights ({weight.shape}) shape"

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        save_pre_act = requires_grad and (act_func is not None)
        output_type = input.dtype
        output = torch.empty(
            (batch_size, out_feat, spatial_dim), device=input.device, dtype=output_type
        )
        pre_act = torch.empty_like(output) if save_pre_act else output

        grid = lambda META: (
            cdiv(out_feat, META["BLOCK_SIZE_M"])
            * cdiv(spatial_dim, META["BLOCK_SIZE_N"]),
            batch_size,
        )
        batched_gemm_kernel[grid](
            # Pointers to matrices
            a_ptr=weight,
            b_ptr=input,
            pre_act_ptr=pre_act,
            c_ptr=output,
            bias_ptr=bias,
            # Matrix dimensions
            M=out_feat,
            N=spatial_dim,
            K=in_feat,
            stride_a_batch=0,
            stride_am=weight.stride(0),
            stride_ak=weight.stride(1),
            stride_b_batch=input.stride(0),
            stride_bk=input.stride(1),
            stride_bn=input.stride(2),
            stride_pre_act_batch=pre_act.stride(0),
            stride_pre_act_m=pre_act.stride(1),
            stride_pre_act_n=pre_act.stride(2),
            stride_c_batch=output.stride(0),
            stride_cm=output.stride(1),
            stride_cn=output.stride(2),
            stride_bias_batch=0,
            stride_bias_feat=bias.stride(0) if bias is not None else 0,
            # precision
            bias_dim=0 if bias is not None else -1,
            fp16=output_type is torch.float16,
            act_param=act_param,
            act_func=act_func,
            save_pre_act=save_pre_act,
        )
        ctx.act_param = act_param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_type = output_type
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(input, weight, pre_act if save_pre_act else None)

        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        batch_size, out_feat, spatial_dim = output_grad.shape

        input, weight, pre_act = ctx.saved_tensors
        if ctx.act_func is None:
            pre_act_grad = output_grad
        else:
            size = batch_size * out_feat * spatial_dim
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad.flatten(),
                pre_act.view_as(pre_act_grad),
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )
            pre_act_grad = pre_act_grad.view_as(output_grad)

        in_feat = weight.shape[1]
        grid_grad_weight = lambda META: (
            cdiv(out_feat, META["BLOCK_SIZE_M"]) * cdiv(in_feat, META["BLOCK_SIZE_N"]),
            batch_size,
        )
        grid_grad_input = lambda META: (
            cdiv(input.shape[1], META["BLOCK_SIZE_M"])
            * cdiv(input.shape[2], META["BLOCK_SIZE_N"]),
            batch_size,
        )
        """
        y = W · X + b
        dL/dW = (dL/dy)·X^T
        dL/dX = W^T·(dL/dy)
        """
        if weight.requires_grad:
            grad_weight = torch.empty(
                (batch_size, out_feat, in_feat),
                device=weight.device,
                dtype=weight.dtype,
            )
            batched_gemm_kernel[grid_grad_weight](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=input,
                pre_act_ptr=None,
                c_ptr=grad_weight,  # [batch_size, out_feat, in_feat]
                bias_ptr=None,
                # Matrix dimensions
                M=out_feat,
                N=in_feat,
                K=spatial_dim,
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(2),
                stride_b_batch=input.stride(0),
                stride_bk=input.stride(2),
                stride_bn=input.stride(1),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_weight.stride(0),
                stride_cm=grad_weight.stride(1),
                stride_cn=grad_weight.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=output_grad.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
            # sum at batch dim
            grad_weight = _sum(grad_weight, dim=0)
        else:
            grad_weight = None
        if input.requires_grad:
            grad_input = torch.empty_like(input, device=input.device, dtype=input.dtype)
            batched_gemm_kernel[grid_grad_input](
                # Pointers to matrices
                a_ptr=weight,
                b_ptr=pre_act_grad,
                pre_act_ptr=None,
                c_ptr=grad_input,
                bias_ptr=None,
                # Matrix dimensions
                M=in_feat,
                N=spatial_dim,
                K=out_feat,
                stride_a_batch=0,
                stride_am=weight.stride(1),
                stride_ak=weight.stride(0),
                stride_b_batch=pre_act_grad.stride(0),
                stride_bk=pre_act_grad.stride(1),
                stride_bn=pre_act_grad.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_input.stride(0),
                stride_cm=grad_input.stride(1),
                stride_cn=grad_input.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=output_grad.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
        else:
            grad_input = None
        grad_bias = pre_act_grad.sum(dim=(0, 2)) if ctx.bias_requires_grad else None
        return grad_input, grad_weight, grad_bias, None


class BatchNormAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        training: bool,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        track_running_stats: bool = True,
        pre_act_add: Optional[Tensor] = None,
        act_func: Optional[str] = None,
    ) -> Tensor:
        # with param in act_func name (e.g., leaky_relu_0.01)
        act_func, act_param = get_act_func(act_func)

        ctx.act_param = act_param
        ctx.act_func = act_func

        # whether add residual
        add_pre_act = pre_act_add is not None
        pre_act_add = (
            pre_act_add if add_pre_act else torch.empty((1, 1, 1), device="cuda")
        )

        input_3d = make_3d_for_bn(input)
        pre_act_add = make_3d_for_bn(pre_act_add)

        # why transpose?
        # transpose = False
        # if input_3d.shape[-1] > 1:
        #     input_3d = input_3d.transpose(0, -1)
        #     pre_act_add = pre_act_add.transpose(0, -1)
        #     transpose = True

        affine = weight is not None and bias is not None
        requires_grad = (
            input.requires_grad
            or pre_act_add.requires_grad
            or (affine and weight.requires_grad)
            or (affine and bias.requires_grad)
        )
        save_pre_act = requires_grad and (act_func is not None)

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)
        pre_act = torch.empty_like(input_3d) if save_pre_act else output

        if requires_grad:
            mean = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
            inv_std = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

        else:
            mean = inv_std = None

        running_mean = input if running_mean is None else running_mean
        running_var = input if running_var is None else running_var

        # Launches 1D grid where each program operates over one feature.
        grid = lambda _: (feat_dim,)
        batch_norm_forward_kernel[grid](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            pre_act_add,  # residual
            pre_act,  # pointer of (x + residual)
            output,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *pre_act_add.stride(),
            *pre_act.stride(),
            *output.stride(),
            momentum,
            eps,
            act_param=act_param,
            affine=affine,
            save_stats=requires_grad,
            track_running_stats=track_running_stats,
            is_train=training,
            add_pre_act=add_pre_act,
            act_func=act_func,
            save_pre_act=save_pre_act,
        )

        # if transpose:
        #     output = output.transpose(0, -1)
        #     if save_pre_act:
        #         pre_act = pre_act.transpose(0, -1)

        ctx.affine = affine
        ctx.act_func = act_func
        ctx.add_pre_act = add_pre_act
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(
                input, mean, inv_std, weight, pre_act if save_pre_act else None
            )

        return output.view_as(input)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        (input, mean, inv_std, weight, pre_act) = ctx.saved_tensors
        input_3d = make_3d_for_bn(input)

        # y = act_func(x)
        # dy = act_func'(x) * dx
        if ctx.act_func is None:
            pre_act_grad = make_3d_for_bn(output_grad)

        else:
            size = output_grad.numel()
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)

            # Launches 1D grid where each program operates over
            # BLOCK_SIZE elements.
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad.flatten(),
                pre_act,
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )

            pre_act_grad = pre_act_grad.view_as(pre_act)

        # transpose = False
        # if input_3d.shape[-1] > 1:
        #     input_3d = input_3d.transpose(0, -1)
        #     pre_act_grad = pre_act_grad.transpose(0, -1)
        #     transpose = True

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        input_grad = torch.empty_like(input_3d)

        if ctx.affine:
            weight_grad = torch.empty((feat_dim,), device=input.device)
            bias_grad = torch.empty_like(weight_grad)

        else:
            weight_grad = bias_grad = None

        # Launches 1D grid where each program operates over one feature.
        grid = lambda _: (feat_dim,)
        batch_norm_backward_kernel[grid](
            pre_act_grad,
            input_3d,
            mean,
            inv_std,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *pre_act_grad.stride(),
            *input_3d.stride(),
            *input_grad.stride(),
            affine=ctx.affine,
        )

        # if transpose:
        #     input_grad = input_grad.transpose(0, -1)
        #     pre_act_grad = pre_act_grad.transpose(0, -1)

        # Pads output with None because a gradient is necessary for
        # all input arguments in `forward`.
        return (
            input_grad.view_as(input),
            None,
            weight_grad,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
            pre_act_grad.view_as(input) if ctx.add_pre_act else None,
            None,
        )


class Conv1d1kAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        act_func: Optional[str] = None,
    ) -> Tensor:
        act_func, act_param = get_act_func(act_func)

        assert input.ndim == 3, f"Input must be 3D, received shape {input.shape}"
        assert weight.ndim == 2, f"Weights must be 2D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), f"Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        out_feat, _ = weight.shape
        batch_size, in_feat, spatial_dim = input.shape

        assert (
            weight.shape[1] == in_feat
        ), f"Incompatible input ({input.shape}) and weights ({weight.shape}) shape"

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        save_pre_act = requires_grad and (act_func is not None)
        output_type = input.dtype
        output = torch.empty(
            (batch_size, out_feat, spatial_dim), device=input.device, dtype=output_type
        )
        pre_act = torch.empty_like(output) if save_pre_act else output

        grid = lambda META: (
            cdiv(out_feat, META["BLOCK_SIZE_M"])
            * cdiv(spatial_dim, META["BLOCK_SIZE_N"]),
            batch_size,
        )
        batched_gemm_kernel[grid](
            # Pointers to matrices
            a_ptr=weight,
            b_ptr=input,
            pre_act_ptr=pre_act,
            c_ptr=output,
            bias_ptr=bias,
            # Matrix dimensions
            M=out_feat,
            N=spatial_dim,
            K=in_feat,
            stride_a_batch=0,
            stride_am=weight.stride(0),
            stride_ak=weight.stride(1),
            stride_b_batch=input.stride(0),
            stride_bk=input.stride(1),
            stride_bn=input.stride(2),
            stride_pre_act_batch=pre_act.stride(0),
            stride_pre_act_m=pre_act.stride(1),
            stride_pre_act_n=pre_act.stride(2),
            stride_c_batch=output.stride(0),
            stride_cm=output.stride(1),
            stride_cn=output.stride(2),
            stride_bias_batch=0,
            stride_bias_feat=bias.stride(0),
            # precision
            bias_dim=0,
            fp16=output_type is torch.float16,
            act_param=act_param,
            act_func=act_func,
            save_pre_act=save_pre_act,
        )
        ctx.act_param = act_param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_type = output_type
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(input, weight, pre_act if save_pre_act else None)

        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        batch_size, out_feat, spatial_dim = output_grad.shape

        input, weight, pre_act = ctx.saved_tensors
        if ctx.act_func is None:
            pre_act_grad = output_grad
        else:
            size = batch_size * out_feat * spatial_dim
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad.flatten(),
                pre_act.view_as(pre_act_grad),
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )
            pre_act_grad = pre_act_grad.view_as(output_grad)

        in_feat = weight.shape[1]
        grid_grad_weight = lambda META: (
            cdiv(out_feat, META["BLOCK_SIZE_M"]) * cdiv(in_feat, META["BLOCK_SIZE_N"]),
            batch_size,
        )
        grid_grad_input = lambda META: (
            cdiv(input.shape[1], META["BLOCK_SIZE_M"])
            * cdiv(input.shape[2], META["BLOCK_SIZE_N"]),
            batch_size,
        )
        """
        y = W · X + b
        dL/dW = (dL/dy)·X^T
        dL/dX = W^T·(dL/dy)
        """
        if weight.requires_grad:
            grad_weight = torch.empty(
                (batch_size, out_feat, in_feat),
                device=weight.device,
                dtype=weight.dtype,
            )
            batched_gemm_kernel[grid_grad_weight](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=input,
                pre_act_ptr=None,
                c_ptr=grad_weight,  # [batch_size, out_feat, in_feat]
                bias_ptr=None,
                # Matrix dimensions
                M=out_feat,
                N=in_feat,
                K=spatial_dim,
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(2),
                stride_b_batch=input.stride(0),
                stride_bk=input.stride(2),
                stride_bn=input.stride(1),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_weight.stride(0),
                stride_cm=grad_weight.stride(1),
                stride_cn=grad_weight.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=output_grad.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
            # sum at batch dim
            grad_weight = grad_weight.sum(dim=0)
        else:
            grad_weight = None
        if input.requires_grad:
            grad_input = torch.empty_like(input, device=input.device, dtype=input.dtype)
            batched_gemm_kernel[grid_grad_input](
                # Pointers to matrices
                a_ptr=weight,
                b_ptr=pre_act_grad,
                pre_act_ptr=None,
                c_ptr=grad_input,
                bias_ptr=None,
                # Matrix dimensions
                M=in_feat,
                N=spatial_dim,
                K=out_feat,
                stride_a_batch=0,
                stride_am=weight.stride(1),
                stride_ak=weight.stride(0),
                stride_b_batch=pre_act_grad.stride(0),
                stride_bk=pre_act_grad.stride(1),
                stride_bn=pre_act_grad.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_input.stride(0),
                stride_cm=grad_input.stride(1),
                stride_cn=grad_input.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=output_grad.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
        else:
            grad_input = None
        grad_bias = pre_act_grad.sum(dim=(0, 2)) if ctx.bias_requires_grad else None
        return grad_input, grad_weight, grad_bias, None


class SoftmaxAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, out: Tensor, log: bool) -> Tensor:
        n_rows, n_cols = input.shape
        """
        对于nums_programs个block,每个block处理一行
        当数据的rows很多的时候,每个block串行的完成rows/num_programs行
        """
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        fwd_kernel, num_programs = softmax_warm_kernels.get(
            str(BLOCK_SIZE) + str(log) + "fwd", (None, 0)
        )
        if fwd_kernel is None:
            fwd_kernel, num_programs = softmax_dim1_warmup(input, log, "fwd")
        num_programs = min(num_programs, n_rows)
        output = out if out is not None else torch.empty_like(input)
        # Create a number of persistent programs.
        """
        这里相当于就已经创建了一个实例
        """
        fwd_kernel[(num_programs, 1, 1)](
            input,  # input_ptr
            output,  # output_ptr
            n_rows,  # n_rows
            n_cols,  # n_cols
            input.stride(0),  # input_stride_row
            input.stride(1),  # input_stride_col
            output.stride(0),  # output_stride_row
            output.stride(1),  # output_stride_col
        )
        ctx.log = log
        require_grad = input.requires_grad
        if require_grad:
            ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Optional[Tensor]:
        (out,) = ctx.saved_tensors
        n_rows, n_cols = out.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        bwd_kernel, num_programs = softmax_warm_kernels.get(
            str(BLOCK_SIZE) + str(ctx.log) + "bwd", (None, 0)
        )
        if bwd_kernel is None:
            bwd_kernel, num_programs = softmax_dim1_warmup(out, ctx.log, "bwd")
        num_programs = min(num_programs, n_rows)
        # Create a number of persistent programs.
        input_grad = torch.empty_like(out)
        bwd_kernel[(num_programs, 1, 1)](
            out,  # output_ptr
            output_grad,  # output_grad_ptr
            input_grad,  # input_grad_ptr
            n_rows,  # n_rows
            n_cols,  # n_cols
            output_grad.stride(0),  # output_grad_stride_row
            output_grad.stride(1),  # output_grad_stride_col
            out.stride(0),  # output_stride_row
            out.stride(1),  # output_stride_col
            input_grad.stride(0),  # input_grad_stride_row
            input_grad.stride(1),  # input_grad_stride_col
        )
        return input_grad, None, None, None


class NLLLossAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        target: Tensor,
        reduction: str = "mean",
        weight: Optional[Tensor] = None,
        output_dtype=torch.float32,
    ) -> Tensor:
        if target.ndim == 1:
            target = target.view(target.shape[0], 1)
        # sanity check
        assert (
            input.ndim >= 2 and input.ndim <= 3 and target.ndim == 2
        ), f"input ({input.shape}) should be 2D or 3D tensor, target ({target.shape}) should be 2D tensor"

        input_3d = input if input.ndim == 3 else input.unsqueeze(-1)

        if weight is not None:
            assert (
                input_3d.shape[1] == weight.shape[0]
            ), f"input_3d ({input_3d.shape}) and weight ({weight.shape}) should have the same feature size"

        assert (
            input_3d.shape[0] == target.shape[0]
        ), f"input_3d ({input_3d.shape}) and target ({target.shape}) should have the same batch size"
        assert (
            input_3d.shape[2] == target.shape[1]
        ), f"input_3d ({input_3d.shape}) and target ({target.shape}) should have the same spatial size"

        assert reduction in [
            "none",
            "mean",
            "sum",
        ], f"reduction should be 'none', 'mean' or 'sum'"
        ## This check is costly
        # assert torch.all(
        #     target < input_3d.shape[1]
        # ), f"All elements in target should be less than the feature size of input_3d ({input_3d.shape[1]})"

        batch_dim, _, spatial_dim = input_3d.shape
        # twice reduce
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic(
            {"batch_dim": batch_dim, "spatial_dim": spatial_dim}
        )
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        output_dtype = output_dtype
        sum_weight = (
            torch.empty(out_batch_dim, dtype=torch.float32, device=input.device)
            if reduction == "mean" and weight is not None
            else None
        )
        output = (
            torch.empty_like(target, dtype=output_dtype)
            if reduction == "none"
            else torch.empty(out_batch_dim, dtype=output_dtype, device=input.device)
        )
        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        # fisrt reduce to out_batch_dim
        grid = (cdiv(batch_dim, BLOCK_SIZE_BATCH),)
        nll_loss_forward_kernel[grid](
            input_ptr=input_3d,
            target_ptr=target,
            weight_ptr=weight,
            output_ptr=output,
            sum_weight_ptr=sum_weight,
            batch_dim=batch_dim,
            spatial_dim=spatial_dim,
            input_batch_stride=input_3d.stride(0),
            input_feat_stride=input_3d.stride(1),
            input_spatial_stride=input_3d.stride(2),
            target_batch_stride=target.stride(0),
            target_spatial_stride=target.stride(1),
            weight_stride=weight.stride(0) if weight is not None else 0,
            output_batch_stride=output.stride(0) if reduction == "none" else 1,
            output_spatial_stride=output.stride(1) if reduction == "none" else 1,
            sum_weight_stride=sum_weight.stride(0) if sum_weight is not None else 1,
            fp16=output_dtype is torch.float16,
            reduction=reduction,
            weighted=weight is not None,
        )
        if reduction != "none":
            output = sum(output, 0)  # scalar

            if reduction == "mean" and weight is not None:
                sum_weight = sum(sum_weight, 0)
                output /= sum_weight

        ctx.sum_weight = sum_weight  # scalar
        ctx.reduction = reduction
        ctx.weight = weight
        ctx.output_dtype = output_dtype
        if input.requires_grad:
            ctx.save_for_backward(input, target)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        sum_weight = ctx.sum_weight
        reduction = ctx.reduction
        weight = ctx.weight
        output_dtype = ctx.output_dtype
        input, target = ctx.saved_tensors
        input_3d = input if input.ndim == 3 else input.unsqueeze(-1)
        batch_dim, _, spatial_dim = input_3d.shape

        grad_input = torch.zeros_like(
            input_3d, dtype=output_dtype, device=grad_output.device
        )
        grad_output = (
            grad_output.view_as(target) if grad_output.ndim > 0 else grad_output
        )
        grid = lambda META: (cdiv(input_3d.shape[0], META["BLOCK_SIZE_BATCH"]),)

        nll_loss_backward_kernel[grid](
            output_grad_ptr=grad_output,
            target_ptr=target,
            weight_ptr=weight,
            sum_weight_ptr=sum_weight,
            input_grad_ptr=grad_input,
            batch_dim=batch_dim,
            spatial_dim=spatial_dim,
            output_grad_batch_stride=(
                grad_output.stride(0) if grad_output.ndim > 0 else 1
            ),
            output_grad_spatial_stride=(
                grad_output.stride(1) if grad_output.ndim > 0 else 1
            ),
            target_batch_stride=target.stride(0),
            target_spatial_stride=target.stride(1),
            weight_stride=weight.stride(0) if weight is not None else 0,
            input_grad_batch_stride=grad_input.stride(0),
            input_grad_feat_stride=grad_input.stride(1),
            input_grad_spatial_stride=grad_input.stride(2),
            fp16=output_dtype is torch.float16,
            reduction=reduction,
            weighted=weight is not None,
        )

        return grad_input.view_as(input), None, None, None, None


### layers
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: Optional[str] = None,
        device: Device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        return LinearAutoGrad.apply(input, self.weight, self.bias, self.act_func)


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func: Optional[str] = None,
        device: Device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.act_func = act_func

    def forward(
        self,
        input: Tensor,
        pre_act_add: Optional[Tensor] = None,
    ) -> Tensor:
        self._check_input_dim(input)

        return BatchNormAutoGrad.apply(
            input,
            self.training,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            self.track_running_stats,
            pre_act_add,
            self.act_func,
        )


class Conv1d1k(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: Optional[str] = None,
        device: Device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        return Conv1d1kAutoGrad.apply(input, self.weight, self.bias, self.act_func)


def bmm(
    a: Tensor,
    b: Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    act_func: Optional[str] = None,
) -> Tensor:
    """
    Function warpper of BatchMatmulAutoGrad.
    """
    return BatchMatmulAutoGrad.apply(a, b, transpose_a, transpose_b, act_func)


def _max(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return MaxReduceAutoGrad.apply(input, dim, keepdim)


def _mean(
    input: Tensor,
    dim: int = 0,
    keepdim: bool = False,
) -> Tensor:
    """
    mean reduce operation warpper
    """
    return MeanReduceAutoGrad.apply(input, dim, keepdim)


def _sum(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return SumReduceAutoGrad.apply(input, dim, keepdim)


def _norm(
    input: Tensor,
    p: float | str | None = 2.0,
    dim: Any | None = -1,
    keepdim: bool = False,
    out: Any | None = None,
    dtype: Any | None = torch.float32,
) -> Tensor:
    """
    p-norm reduce operation warpper
    """
    return NormReduceAutoGrad.apply(input, p, dim, keepdim, out, dtype)


def log_softmax(
    input: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    log: bool = True,
) -> Tensor:
    """
    softmax reduce operation warpper
    """
    if (dim != -1 and dim != 1) or input.dim() != 2:
        raise RuntimeError(f"Only softmax along the last dimension on 2D is supported.")
    softmax_dim1_warmup(input, log)
    return SoftmaxAutoGrad.apply(input, out, log)


def _nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
    output_dtype=torch.float32,
) -> Tensor:
    return NLLLossAutoGrad.apply(input, target, reduction, weight, output_dtype)


### PointNet
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = Conv1d1k(channel, 64)
        self.conv2 = Conv1d1k(64, 128)
        self.conv3 = Conv1d1k(128, 512)
        self.fc1 = Linear(512, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 9)

        self.bn1 = BatchNorm1d(64, act_func="relu")
        self.bn2 = BatchNorm1d(128, act_func="relu")
        self.bn3 = BatchNorm1d(512, act_func="relu")
        self.bn4 = BatchNorm1d(512, act_func="relu")
        self.bn5 = BatchNorm1d(256, act_func="relu")

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        x = _max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = self.bn4(self.fc1(x))
        x = self.bn5(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = Conv1d1k(k, 64, 1)
        self.conv2 = Conv1d1k(64, 128, 1)
        self.conv3 = Conv1d1k(128, 512, 1)
        self.fc1 = Linear(512, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k * k)

        self.bn1 = BatchNorm1d(64, act_func="relu")
        self.bn2 = BatchNorm1d(128, act_func="relu")
        self.bn3 = BatchNorm1d(512, act_func="relu")
        self.bn4 = BatchNorm1d(512, act_func="relu")
        self.bn5 = BatchNorm1d(256, act_func="relu")

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = _max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = self.bn4(self.fc1(x))
        x = self.bn5(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = Conv1d1k(channel, 64, 1)
        self.conv2 = Conv1d1k(64, 128, 1)
        self.conv3 = Conv1d1k(128, 512, 1)
        self.bn1 = BatchNorm1d(64, act_func="relu")
        self.bn2 = BatchNorm1d(128, act_func="relu")
        self.bn3 = BatchNorm1d(512)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = bmm(x, trans, transpose_a=True)
        x = x.transpose(2, 1)
        x = self.bn1(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = bmm(x, trans_feat, transpose_a=True)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = _max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 512, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    y = bmm(trans, trans, transpose_b=True) - I
    y = y.reshape(y.shape[0], -1)
    loss = _mean(_norm(y))
    return loss


# 模型定义
class get_model(nn.Module):
    def __init__(self, k=10, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(
            global_feat=True, feature_transform=True, channel=channel
        )
        self.fc1 = Linear(512, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k)
        self.bn1 = BatchNorm1d(512, act_func="relu")
        self.bn2 = BatchNorm1d(256, act_func="relu")

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.fc3(x)
        x = log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = _nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def save_model_params_and_buffers_to_txt(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 保存所有参数
    for name, param in model.named_parameters():
        np.savetxt(
            os.path.join(directory, f"{name}.txt"),
            param.detach().cpu().numpy().flatten(),
        )

    # 保存所有缓冲区
    for name, buffer in model.named_buffers():
        np.savetxt(
            os.path.join(directory, f"{name}.txt"),
            buffer.detach().cpu().numpy().flatten(),
        )


def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath, "r") as hf:
        for k in hf.keys():
            # list_of_points.append(hf[k]["points"][:].astype(np.float32)) #每个points是（N,3）的二维数组ndarray
            list_of_points.append(
                hf[k]["points"][:].astype(np.float32).flatten()
            )  # 每个points是N*3的一维ndarray
            list_of_labels.append(hf[k].attrs["label"])
    return list_of_points, list_of_labels


class PointCloudDataset(Dataset):
    def __init__(self, root, split, fix_length=128):
        self.list_of_points = []
        self.list_of_labels = []
        self.root = root
        self.split = split
        self.fix_length = fix_length
        # with h5py.File(f"{split}_point_clouds.h5","r") as hf:
        with h5py.File(f"{self.root}/{self.split}_point_clouds.h5", "r") as hf:
            for k in hf.keys():
                self.list_of_points.append(hf[k]["points"][:].astype(np.float32))
                self.list_of_labels.append(hf[k].attrs["label"])
        self.preprocess()

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label

    def data_sample(self, points, fix_length):
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

    def preprocess(self):
        new_list_of_points = []
        for points in self.list_of_points:
            new_list_of_points.append(
                self.data_sample(points=points, fix_length=self.fix_length)
            )
        self.list_of_points = new_list_of_points


### train
def do_train(
    train_dataloader: DataLoader,
    classifier: get_model,
    criterion: get_loss,
    optimizer: Optimizer,
    total_epoch: int = 30,
):
    classifier = classifier.train()
    # 请在本函数下使用triton实现训练操作
    for epoch in range(total_epoch):
        print("Epoch %d (%d/%s):" % (epoch + 1, epoch + 1, total_epoch))
        mean_correct = []
        classifier = classifier.train()

        for batch_id, (points, target) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            points = points.data.numpy()
            # points = random_point_dropout(points)
            # points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)

            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = _max(pred, 1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        print("Train Instance Accuracy: %f" % train_instance_acc)


### infer
def do_inference(classifier: get_model, loader: DataLoader):
    mean_correct = []
    classifier = classifier.eval()

    for _, (points, target) in enumerate(loader):
        points: torch.Tensor = points.cuda()
        target: torch.Tensor = target.cuda()
        points = points.transpose(2, 1)

        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item())

    instance_acc = np.sum(mean_correct) / len(loader.dataset)

    return instance_acc


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 保存模型参数文件(.txt)的文件夹路径
    dir = os.path.abspath(os.path.dirname(__file__)) + "/../param/triton_param"
    # 读取训练集数据
    dataPath = "../data"

    train_dataset = PointCloudDataset(root=dataPath, split="train", fix_length=128)
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=10, drop_last=True
    )
    # model and loss
    criterion = get_loss()
    classifier = get_model(10)
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    # 开始计时
    print("start TRAINING")
    start = time.time()
    do_train(
        train_dataloader=train_dataloader,
        classifier=classifier,
        criterion=criterion,
        optimizer=optimizer,
        total_epoch=30,
    )
    # 结束计时
    end = time.time()
    ms = end - start
    # 保存参数文件，请确保你提交的训练程序可以读取本程序存储的参数文件
    classifier = classifier.eval()
    save_model_params_and_buffers_to_txt(classifier, dir)
    # 输出结果，请严格保持此输出格式，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}")
    test_dataset = PointCloudDataset(root=dataPath, split="test", fix_length=128)
    # 创建 DataLoader 实例
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=False
    )
    """INFERENCE"""
    print("start INFERENCE")
    with torch.no_grad():
        instance_acc = do_inference(classifier, test_dataloader)
        print("Test Instance Accuracy: %f" % (instance_acc))
    print("finish INFERENCE")
