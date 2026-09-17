# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0
"""Shape-specialized RMSNorm kernels for Zing's full-width Q/K tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_ZING_HIDDEN_SIZE = 3072


def is_supported_zing_rmsnorm_hidden_size(hidden_size: int) -> bool:
    return hidden_size == _ZING_HIDDEN_SIZE


@cache_once
def _jit_zing_rmsnorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "diffusion_zing_rmsnorm",
        *args,
        cuda_files=["diffusion/zing_rmsnorm.cuh"],
        cuda_wrappers=[
            ("zing_rmsnorm", f"ZingRMSNormKernel<{args}>::run"),
            ("zing_fused_qknorm", f"ZingFusedQKNormKernel<{args}>::run"),
        ],
    )


def _fake_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    del weight, eps
    return input.new_empty(input.shape)


def _fake_fused_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, eps
    return q.new_empty(q.shape), k.new_empty(k.shape)


@register_custom_op(op_name="zing_rmsnorm", mutates_args=[], fake_impl=_fake_rmsnorm)
def _zing_rmsnorm_custom_op(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if input.numel() == 0:
        return output
    _launch_zing_rmsnorm(input, weight, output, eps)
    return output


@register_custom_op(
    op_name="zing_fused_qknorm",
    mutates_args=[],
    fake_impl=_fake_fused_qknorm,
)
def _zing_fused_qknorm_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_output = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.numel() == 0:
        return q_output, k_output
    _launch_zing_fused_qknorm(q, k, q_weight, k_weight, q_output, k_output, eps)
    return q_output, k_output


def _launch_zing_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> None:
    module = _jit_zing_rmsnorm_module(input.shape[-1], input.dtype)
    module.zing_rmsnorm(input, weight, output, eps)


def _launch_zing_fused_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_output: torch.Tensor,
    k_output: torch.Tensor,
    eps: float,
) -> None:
    module = _jit_zing_rmsnorm_module(q.shape[-1], q.dtype)
    module.zing_fused_qknorm(q, k, q_weight, k_weight, q_output, k_output, eps)


def _flatten_rows(input: torch.Tensor) -> torch.Tensor:
    if input.ndim < 2:
        raise RuntimeError(f"Zing RMSNorm input must be at least 2D, got {input.ndim}D")
    try:
        return input.view(-1, input.shape[-1])
    except RuntimeError as exc:
        raise RuntimeError(
            "Zing RMSNorm requires leading dimensions that flatten without a copy"
        ) from exc


def _validate_input(input: torch.Tensor, weight: torch.Tensor) -> None:
    if not input.is_cuda:
        raise RuntimeError("Zing JIT RMSNorm requires CUDA tensors")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"Zing JIT RMSNorm requires fp16 or bf16 input, got {input.dtype}"
        )
    if input.stride(-1) != 1:
        raise RuntimeError("Zing JIT RMSNorm requires contiguous rows")
    hidden_size = input.shape[-1]
    if not is_supported_zing_rmsnorm_hidden_size(hidden_size):
        raise RuntimeError(
            f"unsupported Zing RMSNorm hidden_size={hidden_size}; "
            f"expected {_ZING_HIDDEN_SIZE}"
        )
    if weight.shape != (hidden_size,):
        raise RuntimeError(
            f"Zing RMSNorm weight must have shape ({hidden_size},), got {tuple(weight.shape)}"
        )
    if weight.device != input.device or weight.dtype != input.dtype:
        raise RuntimeError("Zing RMSNorm weight must match input device and dtype")


def can_use_zing_rmsnorm(input: torch.Tensor, weight: torch.Tensor) -> bool:
    try:
        _validate_input(input, weight)
        _flatten_rows(input)
    except RuntimeError:
        return False
    return True


def zing_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Apply Zing's cast-before-weight RMSNorm and return contiguous output."""
    _validate_input(input, weight)
    shape = input.shape
    flat_input = _flatten_rows(input)
    if torch.compiler.is_compiling():
        output = _zing_rmsnorm_custom_op(flat_input, weight, float(eps))
    else:
        output = torch.empty_like(flat_input)
        if output.numel() != 0:
            _launch_zing_rmsnorm(flat_input, weight, output, float(eps))
    return output.view(shape)


def zing_rmsnorm_unchecked(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Launch the eager fast path after the caller has validated the tensors."""
    shape = input.shape
    flat_input = input.view(-1, _ZING_HIDDEN_SIZE)
    output = torch.empty_like(flat_input)
    if output.numel() != 0:
        _launch_zing_rmsnorm(flat_input, weight, output, float(eps))
    return output.view(shape)


def zing_fused_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize full-width Q/K views in one launch and write contiguous outputs."""
    _validate_input(q, q_weight)
    _validate_input(k, k_weight)
    if q.shape != k.shape:
        raise RuntimeError(
            f"Zing fused Q/K RMSNorm requires equal shapes, got {q.shape} and {k.shape}"
        )
    shape = q.shape
    flat_q = _flatten_rows(q)
    flat_k = _flatten_rows(k)
    if torch.compiler.is_compiling():
        q_output, k_output = _zing_fused_qknorm_custom_op(
            flat_q,
            flat_k,
            q_weight,
            k_weight,
            float(eps),
        )
    else:
        q_output = torch.empty_like(flat_q)
        k_output = torch.empty_like(flat_k)
        if q_output.numel() != 0:
            _launch_zing_fused_qknorm(
                flat_q,
                flat_k,
                q_weight,
                k_weight,
                q_output,
                k_output,
                float(eps),
            )
    return q_output.view(shape), k_output.view(shape)


def zing_fused_qknorm_unchecked(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused eager Q/K RMSNorm after the caller has validated tensors."""
    shape = q.shape
    flat_q = q.view(-1, _ZING_HIDDEN_SIZE)
    flat_k = k.view(-1, _ZING_HIDDEN_SIZE)
    q_output = torch.empty_like(flat_q)
    k_output = torch.empty_like(flat_k)
    if q_output.numel() != 0:
        _launch_zing_fused_qknorm(
            flat_q,
            flat_k,
            q_weight,
            k_weight,
            q_output,
            k_output,
            float(eps),
        )
    return q_output.view(shape), k_output.view(shape)
