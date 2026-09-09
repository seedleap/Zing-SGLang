"""Benchmark MinWM's actual full-width Q/K RMSNorm shapes."""

import statistics
import sys
from collections.abc import Callable

import torch

from sglang.kernels.ops.diffusion.norm.minwm_rmsnorm_jit import (
    _jit_minwm_rmsnorm_module,
    minwm_fused_qknorm,
    minwm_fused_qknorm_unchecked,
    minwm_rmsnorm,
)
from sglang.kernels.ops.layernorm.rmsnorm_hf import _jit_rmsnorm_hf_module, rmsnorm_hf
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone benchmark",
)

EPS = 1e-5
HIDDEN_SIZE = 3072
MAX_ULP_ERROR = 2
# Four latent frames times the post-VAE/post-patch spatial grid, divided by SP.
TOKEN_CASES = (
    ("832x480/SP8", 195),
    ("832x480/SP4", 390),
    ("1248x704/SP8", 429),
    ("1280x704/SP8", 440),
    ("832x480/SP2", 780),
    ("1248x704/SP4", 858),
    ("1280x704/SP4", 880),
    ("832x480/SP1", 1560),
    ("1248x704/SP2", 1716),
    ("1280x704/SP2", 1760),
    ("1248x704/SP1", 3432),
    ("1280x704/SP1", 3520),
)


def _torch_rmsnorm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_float = input.float()
    normalized = input_float * torch.rsqrt(
        input_float.square().mean(dim=-1, keepdim=True) + EPS
    )
    return normalized.to(input.dtype) * weight


def _torch_qknorm_pair(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _torch_rmsnorm(query, query_weight),
        _torch_rmsnorm(key, key_weight),
    )


_torch_compiled_qknorm_pair = torch.compile(
    _torch_qknorm_pair,
    dynamic=True,
    fullgraph=True,
    mode=None,
)


def _time_us(
    function: Callable[[], object], warmup: int = 30, repeats: int = 100
) -> tuple[float, float, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(11):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    samples.sort()
    return statistics.median(samples), samples[1], samples[-2]


def _ordered_16bit_float_bits(input: torch.Tensor) -> torch.Tensor:
    bits = input.view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where((bits & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + magnitude)


def _ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (
        _ordered_16bit_float_bits(actual) - _ordered_16bit_float_bits(expected)
    ).abs()


def _benchmark_shape(num_tokens: int, strided: bool) -> dict[str, float]:
    torch.manual_seed(20260824)
    if strided:
        qkv = torch.randn(
            num_tokens,
            3 * HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        query, key, _ = qkv.chunk(3, dim=-1)
    else:
        query = torch.randn(
            num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn_like(query)
    query_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    minwm_module = _jit_minwm_rmsnorm_module(HIDDEN_SIZE, query.dtype)
    hf_module = _jit_rmsnorm_hf_module(HIDDEN_SIZE, query.dtype)

    def hf_kernel_pair() -> None:
        hf_module.rmsnorm_hf(query, query_weight, query_out, EPS)
        hf_module.rmsnorm_hf(key, key_weight, key_out, EPS)

    def fused_kernel() -> None:
        minwm_module.minwm_fused_qknorm(
            query,
            key,
            query_weight,
            key_weight,
            query_out,
            key_out,
            EPS,
        )

    providers = {
        "torch_eager_pair": lambda: _torch_qknorm_pair(
            query, key, query_weight, key_weight
        ),
        "torch_compile_pair": lambda: _torch_compiled_qknorm_pair(
            query, key, query_weight, key_weight
        ),
        "rmsnorm_hf_pair": lambda: (
            rmsnorm_hf(query, query_weight, EPS),
            rmsnorm_hf(key, key_weight, EPS),
        ),
        "rmsnorm_hf_kernel_pair": hf_kernel_pair,
        "minwm_jit_pair": lambda: (
            minwm_rmsnorm(query, query_weight, EPS),
            minwm_rmsnorm(key, key_weight, EPS),
        ),
        "minwm_jit_fused": lambda: minwm_fused_qknorm(
            query, key, query_weight, key_weight, EPS
        ),
        "minwm_fused_unchecked": lambda: minwm_fused_qknorm_unchecked(
            query, key, query_weight, key_weight, EPS
        ),
        "minwm_fused_kernel": fused_kernel,
    }

    def accuracy(
        candidate: tuple[torch.Tensor, torch.Tensor],
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[int, float]:
        max_ulp = 0
        mismatch_count = 0
        element_count = 0
        for actual, expected in zip(candidate, reference):
            assert torch.isfinite(actual).all().item()
            assert torch.isfinite(expected).all().item()
            max_ulp = max(max_ulp, int(_ulp_distance(actual, expected).max().item()))
            mismatch_count += int(torch.count_nonzero(actual != expected).item())
            element_count += actual.numel()
        return max_ulp, 100.0 * mismatch_count / element_count

    reference = providers["torch_eager_pair"]()
    fused_max_ulp, fused_mismatch_percent = accuracy(
        providers["minwm_jit_fused"](), reference
    )
    compile_max_ulp, compile_mismatch_percent = accuracy(
        providers["torch_compile_pair"](), reference
    )
    assert fused_max_ulp <= MAX_ULP_ERROR

    result = {}
    for name, function in providers.items():
        median_us, p10_us, p90_us = _time_us(function)
        result[name] = median_us
        if name in ("torch_compile_pair", "minwm_fused_unchecked"):
            result[f"{name}_p10"] = p10_us
            result[f"{name}_p90"] = p90_us
    result["fused_max_ulp"] = float(fused_max_ulp)
    result["fused_mismatch_percent"] = fused_mismatch_percent
    result["compile_max_ulp"] = float(compile_max_ulp)
    result["compile_mismatch_percent"] = compile_mismatch_percent
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(
        f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print(
        "| case | tokens | layout | eager us | compile us [p10,p90] | "
        "hf pair us | hf kernel us | jit pair us | fused us | "
        "hot path us [p10,p90] | fused kernel us | vs compile | "
        "fused ULP/mismatch | compile ULP/mismatch |"
    )
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case, num_tokens in TOKEN_CASES:
        for strided in (False, True):
            result = _benchmark_shape(num_tokens, strided)
            speedup = result["torch_compile_pair"] / result["minwm_fused_unchecked"]
            print(
                f"| {case} | {num_tokens} | "
                f"{'stride9216' if strided else 'contiguous'} | "
                f"{result['torch_eager_pair']:.3f} | "
                f"{result['torch_compile_pair']:.3f} "
                f"[{result['torch_compile_pair_p10']:.3f},"
                f"{result['torch_compile_pair_p90']:.3f}] | "
                f"{result['rmsnorm_hf_pair']:.3f} | "
                f"{result['rmsnorm_hf_kernel_pair']:.3f} | "
                f"{result['minwm_jit_pair']:.3f} | {result['minwm_jit_fused']:.3f} | "
                f"{result['minwm_fused_unchecked']:.3f} "
                f"[{result['minwm_fused_unchecked_p10']:.3f},"
                f"{result['minwm_fused_unchecked_p90']:.3f}] | "
                f"{result['minwm_fused_kernel']:.3f} | "
                f"{speedup:.3f}x | "
                f"{result['fused_max_ulp']:.0f}/"
                f"{result['fused_mismatch_percent']:.6f}% | "
                f"{result['compile_max_ulp']:.0f}/"
                f"{result['compile_mismatch_percent']:.6f}% |"
            )


if __name__ == "__main__":
    sys.exit(main())
