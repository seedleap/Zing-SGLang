# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0
"""Hopper-specialized rotary embedding for Zing's interleaved Q/K layout."""

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


_HEAD_DIM = 128
_HALF_HEAD_DIM = _HEAD_DIM // 2


@cache_once
def _is_hopper(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (9, 0)


@cache_once
def _jit_zing_rotary_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        "diffusion_zing_rotary",
        *args,
        cuda_files=["diffusion/zing_rotary.cuh"],
        cuda_wrappers=[
            ("zing_rotary", f"ZingRotaryKernel<{args}>::run"),
        ],
    )


def _fake_zing_rotary(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    del cos, sin, sequence_length
    return hidden_states.new_empty(hidden_states.shape)


@register_custom_op(
    op_name="zing_rotary",
    mutates_args=[],
    fake_impl=_fake_zing_rotary,
)
def _zing_rotary_custom_op(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    output = torch.empty_like(hidden_states)
    if hidden_states.numel() == 0:
        return output
    module = _jit_zing_rotary_module(hidden_states.dtype)
    module.zing_rotary(hidden_states, cos, sin, output, sequence_length)
    return output


@register_custom_op(
    op_name="zing_rotary_out",
    mutates_args=["output"],
)
def _zing_rotary_out_custom_op(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
    sequence_length: int,
) -> None:
    module = _jit_zing_rotary_module(hidden_states.dtype)
    module.zing_rotary(hidden_states, cos, sin, output, sequence_length)


def can_use_zing_rotary(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return whether the exact, vectorized SM90 fast path can consume the inputs."""
    if not hidden_states.is_cuda or hidden_states.ndim < 3:
        return False
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        return False
    if hidden_states.shape[-1] != _HEAD_DIM or not hidden_states.is_contiguous():
        return False
    if hidden_states.numel() == 0:
        return False
    device_index = hidden_states.device.index
    if device_index is None or not _is_hopper(device_index):
        return False

    sequence_length = hidden_states.shape[-3]
    expected_table_elements = sequence_length * _HALF_HEAD_DIM
    for table in (cos, sin):
        if (
            not table.is_cuda
            or table.device != hidden_states.device
            or table.dtype != torch.float32
            or not table.is_contiguous()
            or table.numel() != expected_table_elements
        ):
            return False
    return True


def can_use_zing_rotary_out(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    return (
        can_use_zing_rotary(hidden_states, cos, sin)
        and output.shape == hidden_states.shape
        and output.dtype == hidden_states.dtype
        and output.device == hidden_states.device
        and output.is_contiguous()
    )


def _view_zing_rotary_inputs(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    sequence_length = hidden_states.shape[-3]
    num_heads = hidden_states.shape[-2]
    return (
        hidden_states.view(-1, num_heads, _HEAD_DIM),
        cos.view(sequence_length, _HALF_HEAD_DIM),
        sin.view(sequence_length, _HALF_HEAD_DIM),
        sequence_length,
    )


def zing_rotary(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Zing interleaved RoPE with one SM90 vectorized CUDA kernel."""
    if not can_use_zing_rotary(hidden_states, cos, sin):
        raise RuntimeError("unsupported inputs for Zing Hopper rotary kernel")

    shape = hidden_states.shape
    hidden_3d, cos_2d, sin_2d, sequence_length = _view_zing_rotary_inputs(
        hidden_states, cos, sin
    )
    return _zing_rotary_custom_op(hidden_3d, cos_2d, sin_2d, sequence_length).view(
        shape
    )


def zing_rotary_out(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write exact Zing interleaved RoPE directly into a caller-owned tensor."""
    if not can_use_zing_rotary_out(hidden_states, cos, sin, output):
        raise RuntimeError("unsupported output for Zing Hopper rotary kernel")

    hidden_3d, cos_2d, sin_2d, sequence_length = _view_zing_rotary_inputs(
        hidden_states, cos, sin
    )
    output_3d = output.view_as(hidden_3d)
    _zing_rotary_out_custom_op(hidden_3d, cos_2d, sin_2d, output_3d, sequence_length)
    return output
