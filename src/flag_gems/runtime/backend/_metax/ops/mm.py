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

import functools
import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.device_info import get_l2_cache_size, get_sm_count

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_metax_expand.yaml")
)


def _mm_shape_recording_enabled():
    save_dir = os.getenv("GEMS_SAVE_PATH")
    return bool(
        save_dir
        and os.getenv("USE_GEMS_MODE") == "mm"
        and os.getenv("GEMS_ONCE", "true").lower() == "false"
    )


def _record_mm_shape(func_name, message, *args):
    if not _mm_shape_recording_enabled():
        logger.debug(message, *args)
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    elif os.getenv("RANK", os.getenv("LOCAL_RANK", "0")) != "0":
        return

    save_dir = os.environ["GEMS_SAVE_PATH"]
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "gems-mm.txt")
    line = f"[DEBUG] {__name__}.{func_name}: {message % args}\n"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _prune_mm_dense_configs(configs, named_args, transposed_b=False, **kwargs):
    configs = list(configs)
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    pruned_configs = []

    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        pipeline = config.kwargs["pipeline"]
        scenario = config.kwargs.get("scenario", "")
        warps = config.num_warps
        stages = config.num_stages

        if scenario == "fullstage" and not (
            block_m == 256
            and block_n == 256
            and block_k == 32
            and pipeline == "basic"
            and stages == 2
            and warps == 8
        ):
            continue

        if block_k == 128:
            if block_m > 128 or block_n > 128 or pipeline != "cpasync":
                continue

        if (
            pipeline == "cpasync"
            and block_k >= 64
            and (block_m == 128 or block_n == 128)
        ):
            continue

        if (
            transposed_b
            and pipeline == "cpasync"
            and block_k == 16
            and block_m == 16
            and block_n == 256
            and warps >= 4
        ):
            continue

        if transposed_b and K % block_k != 0 and block_m == 256 and block_n == 256:
            continue

        if (block_m == 128 or block_n == 128) and pipeline == "basic":
            stage_bytes = (block_m + block_n) * block_k * 2 * stages
            if stage_bytes > 64 * 1024:
                continue

        if M >= 1024 and N >= 128:
            if block_m not in (64, 128, 256) or block_n not in (64, 128, 256):
                continue
            if warps not in (4, 8):
                continue
        else:
            if block_m == 128 or warps == 8:
                continue
            if N <= 64 and (block_m > 64 or block_n > 64):
                continue
            if warps == 2 and not (block_m <= 32 and block_n <= 64):
                continue

        pruned_configs.append(config)

    return pruned_configs or configs


_prune_mm_dense_configs_nt = functools.partial(
    _prune_mm_dense_configs, transposed_b=True
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
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
    else:
        pid = tl.program_id(0)
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

    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc = tl.dot(
            a,
            b,
            acc,
            out_dtype=dot_out_dtype,
            allow_tf32=False,
        )
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
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
    if EVEN_M and EVEN_N:
        tl.store(C, acc)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_nn_bf16"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nn_bf16",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def mm_kernel_nn_bf16(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rbn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    result = acc.to(tl.bfloat16)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_nt_bf16"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs_nt},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nt_bf16",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def mm_kernel_nt_bf16(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] + rbn[None, :] * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    result = acc.to(tl.bfloat16)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


def _prune_gemv_configs(configs, named_args, **kwargs):
    configs = list(configs)
    pruned_configs = [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] == 256 and config.num_warps in (4, 8)
    ]
    return pruned_configs or configs


@libentry()
@libtuner(
    configs=[triton.Config({"BLOCK_M": 32, "BLOCK_K": 256})],
    key=["M", "K", "stride_am", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
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
    _record_mm_shape(
        "gemv_mm",
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


def _reset_splitk_output(args, reset_only=False):
    # Triton benchmarks with positional args; LibTuner resets cached configs by name.
    c = args["C"] if isinstance(args, dict) else args[2]
    c.zero_()


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_splitk"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    pre_hook=_reset_splitk_output,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_splitk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
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
    _record_mm_shape(
        "splitk_mm",
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
    _record_mm_shape(
        "general_mm",
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
    )
    with torch_device_fn.device(a.device):
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


def general_mm_nn_bf16(a, b, c, M, N, K):
    _record_mm_shape(
        "general_mm_nn_bf16",
        "GEMS_METAX MM, [mm scenario]: general_nn_bf16, [shape info]: "
        "[-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_nn_bf16[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            GROUP_M=8,
        )
    return c


def general_mm_nt_bf16(a, b, c, M, N, K):
    _record_mm_shape(
        "general_mm_nt_bf16",
        "GEMS_METAX MM, [mm scenario]: general_nt_bf16, [shape info]: "
        "[-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_nt_bf16[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            GROUP_M=8,
        )
    return c


@functools.lru_cache(maxsize=1)
def _dense_output_tile_candidates():
    tile_candidates = set()
    for config_name in ("mm", "mm_nn_bf16", "mm_nt_bf16"):
        tile_candidates.update(
            (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"])
            for config in runtime.get_tuned_config(config_name)
            if "BLOCK_M" in config.kwargs and "BLOCK_N" in config.kwargs
        )

        expand_config = runtime.get_expand_config(
            config_name,
            yaml_path=EXPAND_CONFIG_FILENAME,
        )
        if expand_config != -1:
            ranges = expand_config["ranges"]
            block_ms = ranges.get("BLOCK_M", ())
            block_ns = ranges.get("BLOCK_N", ())
            tile_candidates.update(
                (block_m, block_n)
                for block_m in block_ms
                for block_n in block_ns
            )

    return tuple(sorted(tile_candidates))


def _max_general_mm_programs(M, N):
    tile_candidates = _dense_output_tile_candidates()
    if not tile_candidates:
        return get_sm_count()

    return max(
        triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        for block_m, block_n in tile_candidates
    )


def splitk_mm_scenario(M, N, K):
    return M < 1024 and N < 1024 and K >= 2048 and M * N < 32768


def nn_bf16_mm_scenario(a, b, c, M, N, K):
    return (
        M > 0
        and N > 0
        and K > 0
        and a.dtype is torch.bfloat16
        and b.dtype is torch.bfloat16
        and c.dtype is torch.bfloat16
        and a.stride(0) == K
        and a.stride(1) == 1
        and b.stride(0) == N
        and b.stride(1) == 1
        and c.stride(0) == N
        and c.stride(1) == 1
        and M * K < 2**31
        and K * N < 2**31
        and M * N < 2**31
    )


def mm(a, b):
    if not _mm_shape_recording_enabled():
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
    if splitk_mm_scenario(M, N, K):
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
    if nn_bf16_mm_scenario(a, b, c, M, N, K):
        return general_mm_nn_bf16(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    if not _mm_shape_recording_enabled():
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
    if splitk_mm_scenario(M, N, K):
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
    if nn_bf16_mm_scenario(a, b, c, M, N, K):
        return general_mm_nn_bf16(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)
