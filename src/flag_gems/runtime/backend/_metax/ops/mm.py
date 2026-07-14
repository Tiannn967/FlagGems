# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.heuristics(
    {
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = ext.program_id(0)
        pid_z = ext.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        if EVEN_M:
            ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M).to(
                tl.int64
            )
        else:
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(
                tl.int64
            )
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if EVEN_M:
            ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        else:
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        if EVEN_N:
            rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N).to(
                tl.int64
            )
        else:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(
                tl.int64
            )
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if EVEN_N:
            rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        else:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        if EVEN_M and EVEN_N:
            tl.store(C, acc)
        else:
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C, acc, mask=mask)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.heuristics(
    {
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit
def mm_kernel_contiguous(
    A,
    B,
    C,
    M,
    N,
    K,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    if UPGRADE:
        pid = ext.program_id(0)
        pid_z = ext.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        if EVEN_M:
            ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M).to(
                tl.int64
            )
        else:
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(
                tl.int64
            )
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if EVEN_M:
            ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        else:
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        if EVEN_N:
            rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N).to(
                tl.int64
            )
        else:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(
                tl.int64
            )
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if EVEN_N:
            rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        else:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + ram[:, None] * K + rk[None, :]
    B = B + rk[:, None] * N + rbn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K
        B += BLOCK_K * SPLIT_K * N

    acc = acc.to(C.dtype.element_ty)
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * N + rn[None, :]).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + rm[:, None] * N + rn[None, :]
    if SPLIT_K == 1:
        if EVEN_M and EVEN_N:
            tl.store(C, acc)
        else:
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C, acc, mask=mask)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=[triton.Config({"BLOCK_M": 32, "BLOCK_K": 256})],
    key=["M", "K", "stride_am", "stride_bk"],
)
@triton.jit
def gemv_kernel(
    A,
    B,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K
        a = tl.load(
            A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(B + offsets_k * stride_bk, mask=mask_k, other=0.0)
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    tl.store(C + offsets_m * stride_cm, acc.to(C.dtype.element_ty), mask=mask_m)


def gemv_mm(a, b, c, M, K):
    logger.debug(
        "GEMS_METAX MM, [mm scenario]: gemv (N=1), [shape info]: [%s, %s, 1](M, K, N)",
        M,
        K,
    )
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(a.device):
        gemv_kernel[grid](
            a,
            b,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            c.stride(0),
        )
    return c


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_splitk"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    reset_to_zero=["C"],
)
@triton.jit
def mm_kernel_splitk(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end):
        offsets_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    mask = (offsets_m < M)[:, None] & (offsets_n < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)


def splitk_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS_METAX MM, [mm scenario]: splitk, [shape info]: "
        "[-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_kernel_splitk[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
    return c


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def general_mm(a, b, c, M, N, K):
    dot_out_dtype = tl.float32
    logger.debug(
        "GEMS_METAX MM, [mm scenario]: general, [shape info]: "
        "[-, %s, %s, %s](batch, M, N, K), [A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        if a.is_contiguous() and b.is_contiguous() and c.is_contiguous():
            mm_kernel_contiguous[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                dot_out_dtype=dot_out_dtype,
                GROUP_M=8,
            )
        else:
            mm_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                dot_out_dtype=dot_out_dtype,
                GROUP_M=8,
            )
    return c


def mm(a, b):
    logger.debug("GEMS_METAX MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    if M < 2048 and N < 2048 and K >= 4096:
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS_METAX MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    if M < 2048 and N < 2048 and K >= 4096:
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)
