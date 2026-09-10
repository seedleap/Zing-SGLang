import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.rope.minwm_rotary_jit import (
    can_use_minwm_rotary,
    can_use_minwm_rotary_out,
    minwm_rotary,
    minwm_rotary_out,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


TOKENS_PER_CHUNK = (1560, 3432, 3520)
PRODUCTION_SEQUENCE_LENGTHS = tuple(
    sorted(
        chunk_tokens * cached_chunks
        for chunk_tokens in TOKENS_PER_CHUNK
        for cached_chunks in range(1, 9)
    )
)
PRODUCTION_NUM_HEADS = (6, 12, 24)
BOUNDARY_CASES = (
    (1, 1),
    (15, 3),
    (16, 5),
    (17, 7),
    (31, 8),
    (32, 23),
    (33, 25),
)
RESOLUTION_TAIL_AND_LONG_CASES = tuple(
    (sequence_length, num_heads)
    for chunk_tokens in TOKENS_PER_CHUNK
    for sequence_length, num_heads in (
        (chunk_tokens - 1, 6),
        (chunk_tokens + 1, 12),
        (chunk_tokens * 8 - 1, 24),
        (chunk_tokens * 8 + 1, 6),
        (chunk_tokens * 9, 12),
        (chunk_tokens * 12, 24),
    )
)


def _torch_reference(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    sequence_length = hidden_states.shape[-3]
    half_head_dim = hidden_states.shape[-1] // 2
    broadcast_shape = (1,) * (hidden_states.dim() - 3) + (
        sequence_length,
        1,
        half_head_dim,
    )
    real = hidden_states[..., 0::2].float()
    imaginary = hidden_states[..., 1::2].float()
    cos = cos.reshape(broadcast_shape)
    sin = sin.reshape(broadcast_shape)
    return (
        torch.stack(
            (real * cos - imaginary * sin, real * sin + imaginary * cos),
            dim=-1,
        )
        .flatten(-2)
        .type_as(hidden_states)
    )


@pytest.fixture(autouse=True)
def hopper_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("MinWM rotary fast path targets Hopper SM90")
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("sequence_length", PRODUCTION_SEQUENCE_LENGTHS)
@pytest.mark.parametrize("num_heads", PRODUCTION_NUM_HEADS)
def test_minwm_rotary_matches_production_bf16_shapes(sequence_length, num_heads):
    hidden = torch.randn(
        1,
        sequence_length,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    angles = torch.randn(sequence_length, 64, device="cuda", dtype=torch.float32)
    cos = angles.cos()
    sin = angles.sin()

    assert can_use_minwm_rotary(hidden, cos, sin)
    actual = minwm_rotary(hidden, cos, sin)
    expected = _torch_reference(hidden, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("sequence_length", "num_heads"),
    [(5, 3), (1560, 6), (28160, 24)],
)
def test_minwm_rotary_matches_explicit_fp32_formula_fp16(sequence_length, num_heads):
    hidden = torch.randn(
        1,
        sequence_length,
        num_heads,
        128,
        device="cuda",
        dtype=torch.float16,
    )
    angles = torch.randn(sequence_length, 64, device="cuda", dtype=torch.float32)
    cos = angles.cos()
    sin = angles.sin()

    assert can_use_minwm_rotary(hidden, cos, sin)
    actual = minwm_rotary(hidden, cos, sin)
    expected = _torch_reference(hidden, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    ("sequence_length", "num_heads"),
    BOUNDARY_CASES + RESOLUTION_TAIL_AND_LONG_CASES,
)
def test_minwm_rotary_matches_boundary_tail_and_long_shapes(
    dtype, sequence_length, num_heads
):
    hidden = torch.randn(
        1,
        sequence_length,
        num_heads,
        128,
        device="cuda",
        dtype=dtype,
    )
    angles = torch.randn(sequence_length, 64, device="cuda", dtype=torch.float32)
    cos = angles.cos()
    sin = angles.sin()

    assert can_use_minwm_rotary(hidden, cos, sin)
    actual = minwm_rotary(hidden, cos, sin)
    expected = _torch_reference(hidden, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    (
        (2, 17, 5, 128),
        (2, 3, 33, 7, 128),
        (2, 1560, 6, 128),
    ),
)
def test_minwm_rotary_matches_multi_batch_prefixes(shape):
    hidden = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    sequence_length = shape[-3]
    angles = torch.randn(sequence_length, 64, device="cuda", dtype=torch.float32)
    cos = angles.cos()
    sin = angles.sin()

    assert can_use_minwm_rotary(hidden, cos, sin)
    actual = minwm_rotary(hidden, cos, sin)
    expected = _torch_reference(hidden, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("sequence_length", "num_heads"),
    [(1560, 6), (3432, 12), (28160, 24)],
)
def test_minwm_rotary_out_matches_explicit_fp32_formula(sequence_length, num_heads):
    hidden = torch.randn(
        1,
        sequence_length,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    angles = torch.randn(sequence_length, 64, device="cuda", dtype=torch.float32)
    cos = angles.cos()
    sin = angles.sin()
    output = torch.empty_like(hidden)

    assert can_use_minwm_rotary_out(hidden, cos, sin, output)
    actual = minwm_rotary_out(hidden, cos, sin, output)
    expected = _torch_reference(hidden, cos, sin)
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_minwm_rotary_rejects_noncontiguous_tables():
    hidden = torch.randn(1, 5, 24, 128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(5, 128, device="cuda", dtype=torch.float32)
    cos = angles[:, 0::2]
    sin = angles[:, 1::2]

    assert not can_use_minwm_rotary(hidden, cos, sin)


def test_minwm_rotary_custom_op_torch_compile_fullgraph():
    hidden = torch.randn(1, 32, 24, 128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    cos = angles.cos()
    sin = angles.sin()

    compiled = torch.compile(minwm_rotary, fullgraph=True)
    actual = compiled(hidden, cos, sin)
    expected = _torch_reference(hidden, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
