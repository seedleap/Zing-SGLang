import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.norm.zing_rmsnorm_jit import (
    can_use_minwm_rmsnorm,
    is_supported_minwm_rmsnorm_hidden_size,
    minwm_fused_qknorm,
    minwm_fused_qknorm_unchecked,
    minwm_rmsnorm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

EPS = 1e-5
HIDDEN_SIZE = 3072
MAX_ULP_ERROR = 2
# Four latent frames times the post-VAE/post-patch spatial grid, divided by SP:
# 832x480: 4 * 15 * 26 = 1560; 1248x704: 4 * 22 * 39 = 3432;
# 1280x704: 4 * 22 * 40 = 3520.
SELF_QK_TOKEN_CASES = (
    ("832x480-SP8", 195),
    ("832x480-SP4", 390),
    ("1248x704-SP8", 429),
    ("1280x704-SP8", 440),
    ("832x480-SP2", 780),
    ("1248x704-SP4", 858),
    ("1280x704-SP4", 880),
    ("832x480-SP1", 1560),
    ("1248x704-SP2", 1716),
    ("1280x704-SP2", 1760),
    ("1248x704-SP1", 3432),
    ("1280x704-SP1", 3520),
)


def _reference(
    input: torch.Tensor, weight: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    input_float = input.float()
    normalized = input_float * torch.rsqrt(
        input_float.square().mean(dim=-1, keepdim=True) + eps
    )
    return normalized.to(input.dtype) * weight


def _ordered_16bit_float_bits(input: torch.Tensor) -> torch.Tensor:
    bits = input.view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where((bits & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + magnitude)


def _ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (
        _ordered_16bit_float_bits(actual) - _ordered_16bit_float_bits(expected)
    ).abs()


def _assert_strict_numerics(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.dtype in (torch.float16, torch.bfloat16)
    assert torch.isfinite(actual).all().item()
    assert torch.isfinite(expected).all().item()
    max_ulp = int(_ulp_distance(actual, expected).max().item())
    assert max_ulp <= MAX_ULP_ERROR, f"max ULP error {max_ulp} > {MAX_ULP_ERROR}"


def _make_strided_qk(
    num_tokens: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    qkv = torch.randn(num_tokens, 3 * HIDDEN_SIZE, device="cuda", dtype=dtype)
    query, key, _ = qkv.chunk(3, dim=-1)
    assert query.stride() == (3 * HIDDEN_SIZE, 1)
    return query, key


@pytest.mark.parametrize(
    ("shape_name", "num_tokens"),
    (("single-row", 1), *SELF_QK_TOKEN_CASES),
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_minwm_fused_qknorm_matches_round_before_weight_reference(
    shape_name: str, num_tokens: int, dtype: torch.dtype
) -> None:
    del shape_name
    torch.manual_seed(20260824)
    query, key = _make_strided_qk(num_tokens, dtype)
    query_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=dtype)
    key_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=dtype)

    query_out, key_out = minwm_fused_qknorm(query, key, query_weight, key_weight, EPS)
    unchecked_query_out, unchecked_key_out = minwm_fused_qknorm_unchecked(
        query, key, query_weight, key_weight, EPS
    )

    assert query_out.is_contiguous()
    assert key_out.is_contiguous()
    _assert_strict_numerics(query_out, _reference(query, query_weight))
    _assert_strict_numerics(key_out, _reference(key, key_weight))
    assert torch.equal(unchecked_query_out, query_out)
    assert torch.equal(unchecked_key_out, key_out)


def test_minwm_rmsnorm_covers_cross_attention_shape() -> None:
    torch.manual_seed(20260824)
    query = torch.randn(1, 512, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    output = minwm_rmsnorm(query, weight, EPS)

    assert output.shape == query.shape
    assert output.is_contiguous()
    _assert_strict_numerics(output, _reference(query, weight))


def test_minwm_rmsnorm_uses_cast_before_weight_multiply() -> None:
    torch.manual_seed(20260824)
    input = torch.randn(64, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    output = minwm_rmsnorm(input, weight, EPS)
    reference = _reference(input, weight)
    input_float = input.float()
    multiply_before_rounding = (
        input_float
        * torch.rsqrt(input_float.square().mean(dim=-1, keepdim=True) + EPS)
        * weight.float()
    ).to(input.dtype)

    _assert_strict_numerics(output, reference)
    reference_distance = _ulp_distance(output, reference).sum().item()
    wrong_order_distance = _ulp_distance(output, multiply_before_rounding).sum().item()
    assert not torch.equal(reference, multiply_before_rounding)
    assert reference_distance < wrong_order_distance


def test_minwm_fused_qknorm_torch_compile_fullgraph() -> None:
    query, key = _make_strided_qk(195, torch.bfloat16)
    query_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    compiled = torch.compile(minwm_fused_qknorm, fullgraph=True)
    query_out, key_out = compiled(query, key, query_weight, key_weight, EPS)

    _assert_strict_numerics(query_out, _reference(query, query_weight))
    _assert_strict_numerics(key_out, _reference(key, key_weight))


def test_minwm_model_qk_norm_integration() -> None:
    from sglang.multimodal_gen.runtime.models.dits.zing import _minwm_qk_norm_op

    query, key = _make_strided_qk(195, torch.bfloat16)
    query_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    query_out, key_out = _minwm_qk_norm_op(
        query, key, query_weight, key_weight, EPS, num_heads=24
    )

    assert query_out.shape == (195, 24, 128)
    assert key_out.shape == (195, 24, 128)
    _assert_strict_numerics(query_out.flatten(-2), _reference(query, query_weight))
    _assert_strict_numerics(key_out.flatten(-2), _reference(key, key_weight))


def test_minwm_rmsnorm_support_gate() -> None:
    assert is_supported_minwm_rmsnorm_hidden_size(3072)
    assert not is_supported_minwm_rmsnorm_hidden_size(128)
    assert not is_supported_minwm_rmsnorm_hidden_size(3073)

    cpu_input = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16)
    cpu_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
    assert not can_use_minwm_rmsnorm(cpu_input, cpu_weight)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
