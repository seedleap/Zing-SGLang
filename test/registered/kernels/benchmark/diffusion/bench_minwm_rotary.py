import argparse
import inspect
import random
import sys
import time
from dataclasses import dataclass
from functools import partial

import torch

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except Exception:
    apply_rope_with_cos_sin_cache_inplace = None

from sglang.kernels.ops.diffusion.rope.minwm_rotary_jit import (
    minwm_rotary,
    minwm_rotary_out,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@dataclass(frozen=True)
class Workload:
    name: str
    sequence_length: int
    num_heads: int
    phase: str


TOKENS_PER_CHUNK = {
    "480p_832x480": 1560,
    "720p_1248x704": 3432,
    "720p_1280x704": 3520,
}
SP_HEADS = {"sp1": 24, "sp2": 12, "sp4": 6}
WORKLOADS = [
    Workload(f"{resolution}_{phase}_{sp}", tokens * cache_chunks, num_heads, phase)
    for resolution, tokens in TOKENS_PER_CHUNK.items()
    for phase, cache_chunks in (("query", 1), ("kv_cache", 8))
    for sp, num_heads in SP_HEADS.items()
]


def torch_reference(hidden, cos, sin):
    real = hidden[..., 0::2].float()
    imaginary = hidden[..., 1::2].float()
    cos = cos.view(1, hidden.shape[-3], 1, 64)
    sin = sin.view(1, hidden.shape[-3], 1, 64)
    return (
        torch.stack(
            (real * cos - imaginary * sin, real * sin + imaginary * cos),
            dim=-1,
        )
        .flatten(-2)
        .to(hidden.dtype)
    )


def torch_reference_out(hidden, cos, sin, output):
    output.copy_(torch_reference(hidden, cos, sin))
    return output


def cuda_event_us(fn, warmups=20, repeats=20, rounds=11):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    samples.sort()
    return samples[len(samples) // 2]


def host_dispatch_us(fn, warmups=5, rounds=31):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = fn()
        elapsed = time.perf_counter_ns() - start
        del output
        torch.cuda.synchronize()
        samples.append(elapsed / 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def peak_extra_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    torch.cuda.synchronize()
    return peak / (1024**2)


def compile_first_call(fn, *args, mode=None):
    kwargs = {}
    if "recompile_limit" in inspect.signature(torch.compile).parameters:
        kwargs["recompile_limit"] = 64
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, mode=mode, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = compiled(*args)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter_ns() - start) / 1e6
    return compiled, output, first_call_ms


def mismatch_stats(actual, expected):
    mismatches = int(torch.count_nonzero(actual != expected))
    max_abs_error = float((actual.float() - expected.float()).abs().max())
    return mismatches, max_abs_error


def benchmark_torch_compile(compile_mode=None):
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    if torch.cuda.get_device_capability() != (9, 0):
        print("Hopper SM90 required")
        return

    torch.manual_seed(20260825)
    random.seed(20260825)
    print()
    print(
        "torch.compile mode: fullgraph=True, dynamic=False, "
        f"backend=inductor, mode={compile_mode or 'default'}"
    )
    print(
        "| workload | shape | compiled eager GPU us | hopper GPU us | "
        "compiled hopper GPU us | hopper speedup vs compiled eager | compile-on-hopper "
        "GPU delta |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    host_results = []
    out_results = []

    for workload in WORKLOADS:
        hidden = torch.randn(
            1,
            workload.sequence_length,
            workload.num_heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        angles = torch.randn(
            workload.sequence_length, 64, device="cuda", dtype=torch.float32
        )
        cos = angles.cos()
        sin = angles.sin()
        expected = torch_reference(hidden, cos, sin)

        compiled_eager, compiled_actual, eager_first_ms = compile_first_call(
            torch_reference, hidden, cos, sin, mode=compile_mode
        )
        compiled_mismatches, compiled_max_abs_error = mismatch_stats(
            compiled_actual, expected
        )
        torch.testing.assert_close(compiled_actual, expected, rtol=1e-2, atol=1e-2)

        compiled_hopper, compiled_hopper_actual, hopper_first_ms = compile_first_call(
            minwm_rotary, hidden, cos, sin, mode=compile_mode
        )
        torch.testing.assert_close(compiled_hopper_actual, expected, rtol=0, atol=0)

        fns = {
            "compiled_eager": partial(compiled_eager, hidden, cos, sin),
            "hopper": partial(minwm_rotary, hidden, cos, sin),
            "compiled_hopper": partial(compiled_hopper, hidden, cos, sin),
        }
        order = list(fns)
        random.shuffle(order)
        times = {name: cuda_event_us(fns[name]) for name in order}
        host_times = {name: host_dispatch_us(fns[name]) for name in order}

        eager_out = torch.empty_like(hidden)
        hopper_out = torch.empty_like(hidden)
        compiled_hopper_out = torch.empty_like(hidden)
        compiled_eager_out, compiled_eager_out_actual, eager_out_first_ms = (
            compile_first_call(
                torch_reference_out,
                hidden,
                cos,
                sin,
                eager_out,
                mode=compile_mode,
            )
        )
        out_mismatches, out_max_abs_error = mismatch_stats(
            compiled_eager_out_actual, expected
        )
        torch.testing.assert_close(
            compiled_eager_out_actual, expected, rtol=1e-2, atol=1e-2
        )
        compiled_hopper_out_fn, compiled_hopper_out_actual, hopper_out_first_ms = (
            compile_first_call(
                minwm_rotary_out,
                hidden,
                cos,
                sin,
                compiled_hopper_out,
                mode=compile_mode,
            )
        )
        torch.testing.assert_close(compiled_hopper_out_actual, expected, rtol=0, atol=0)

        out_fns = {
            "compiled_eager": partial(compiled_eager_out, hidden, cos, sin, eager_out),
            "hopper": partial(minwm_rotary_out, hidden, cos, sin, hopper_out),
            "compiled_hopper": partial(
                compiled_hopper_out_fn, hidden, cos, sin, compiled_hopper_out
            ),
        }
        out_order = list(out_fns)
        random.shuffle(out_order)
        out_times = {name: cuda_event_us(out_fns[name]) for name in out_order}
        out_host_times = {name: host_dispatch_us(out_fns[name]) for name in out_order}

        shape = f"[1,{workload.sequence_length},{workload.num_heads},128]"
        print(
            f"| {workload.name} | `{shape}` | {times['compiled_eager']:.2f} | "
            f"{times['hopper']:.2f} | {times['compiled_hopper']:.2f} | "
            f"{times['compiled_eager'] / times['hopper']:.2f}x | "
            f"{(times['compiled_hopper'] / times['hopper'] - 1.0) * 100:+.1f}% |"
        )
        host_results.append(
            (
                workload.name,
                shape,
                host_times,
                eager_first_ms,
                hopper_first_ms,
                compiled_mismatches,
                compiled_max_abs_error,
            )
        )
        out_results.append(
            (
                workload.name,
                shape,
                out_times,
                out_host_times,
                eager_out_first_ms,
                hopper_out_first_ms,
                out_mismatches,
                out_max_abs_error,
            )
        )
        del (
            compiled_eager,
            compiled_actual,
            compiled_hopper,
            compiled_hopper_actual,
            compiled_eager_out,
            compiled_eager_out_actual,
            compiled_hopper_out_fn,
            compiled_hopper_out_actual,
            expected,
            hidden,
            angles,
            cos,
            sin,
            eager_out,
            hopper_out,
            compiled_hopper_out,
        )
        torch.cuda.empty_cache()

    print()
    print(
        "| workload | compiled eager host us | hopper host us | compiled hopper "
        "host us | eager first call ms | hopper first call ms | compiled eager "
        "mismatches | max abs error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for (
        name,
        _shape,
        host_times,
        eager_first_ms,
        hopper_first_ms,
        mismatches,
        max_abs_error,
    ) in host_results:
        print(
            f"| {name} | {host_times['compiled_eager']:.2f} | "
            f"{host_times['hopper']:.2f} | {host_times['compiled_hopper']:.2f} | "
            f"{eager_first_ms:.2f} | {hopper_first_ms:.2f} | {mismatches} | "
            f"{max_abs_error:.8f} |"
        )

    print()
    print(
        "| workload | shape | compiled eager direct-out GPU us | hopper direct-out "
        "GPU us | compiled hopper direct-out GPU us | hopper speedup vs compiled eager | "
        "compile-on-hopper GPU delta | compiled eager mismatches | max abs error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (
        name,
        shape,
        times,
        _host_times,
        _eager_first_ms,
        _hopper_first_ms,
        mismatches,
        max_abs_error,
    ) in out_results:
        print(
            f"| {name} | `{shape}` | {times['compiled_eager']:.2f} | "
            f"{times['hopper']:.2f} | {times['compiled_hopper']:.2f} | "
            f"{times['compiled_eager'] / times['hopper']:.2f}x | "
            f"{(times['compiled_hopper'] / times['hopper'] - 1.0) * 100:+.1f}% | "
            f"{mismatches} | {max_abs_error:.8f} |"
        )

    print()
    print(
        "| workload | compiled eager direct-out host us | hopper direct-out host us | "
        "compiled hopper direct-out host us | eager first call ms | hopper first "
        "call ms |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for (
        name,
        _shape,
        _times,
        host_times,
        eager_first_ms,
        hopper_first_ms,
        _mismatches,
        _max_abs_error,
    ) in out_results:
        print(
            f"| {name} | {host_times['compiled_eager']:.2f} | "
            f"{host_times['hopper']:.2f} | {host_times['compiled_hopper']:.2f} | "
            f"{eager_first_ms:.2f} | {hopper_first_ms:.2f} |"
        )


def benchmark():
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    if torch.cuda.get_device_capability() != (9, 0):
        print("Hopper SM90 required")
        return

    torch.manual_seed(20260824)
    random.seed(20260824)
    print(
        "| workload | shape | torch GPU us | hopper GPU us | speedup | "
        "torch host us | hopper host us | torch peak MiB | hopper peak MiB |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    cache_store_results = []
    flashinfer_results = []

    for workload in WORKLOADS:
        hidden = torch.randn(
            1,
            workload.sequence_length,
            workload.num_heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        angles = torch.randn(
            workload.sequence_length, 64, device="cuda", dtype=torch.float32
        )
        cos = angles.cos()
        sin = angles.sin()

        expected = torch_reference(hidden, cos, sin)
        actual = minwm_rotary(hidden, cos, sin)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del actual, expected
        torch.cuda.synchronize()

        fns = {
            "torch": lambda: torch_reference(hidden, cos, sin),
            "hopper": lambda: minwm_rotary(hidden, cos, sin),
        }
        order = list(fns)
        random.shuffle(order)
        times = {name: cuda_event_us(fns[name]) for name in order}
        host_times = {name: host_dispatch_us(fns[name]) for name in order}
        peak_memory = {name: peak_extra_mib(fns[name]) for name in order}
        cache_output = torch.empty_like(hidden)
        store_fns = {
            "copy": lambda: cache_output.copy_(minwm_rotary(hidden, cos, sin)),
            "out": lambda: minwm_rotary_out(hidden, cos, sin, cache_output),
        }
        store_order = list(store_fns)
        random.shuffle(store_order)
        store_times = {name: cuda_event_us(store_fns[name]) for name in store_order}
        store_host_times = {
            name: host_dispatch_us(store_fns[name]) for name in store_order
        }
        store_peak_memory = {
            name: peak_extra_mib(store_fns[name]) for name in store_order
        }
        shape = f"[1,{workload.sequence_length},{workload.num_heads},128]"
        print(
            f"| {workload.name} | `{shape}` | {times['torch']:.2f} | "
            f"{times['hopper']:.2f} | {times['torch'] / times['hopper']:.2f}x | "
            f"{host_times['torch']:.2f} | {host_times['hopper']:.2f} | "
            f"{peak_memory['torch']:.2f} | {peak_memory['hopper']:.2f} |"
        )
        cache_store_results.append(
            (
                workload.name,
                shape,
                store_times,
                store_host_times,
                store_peak_memory,
            )
        )
        if workload.phase == "query" and apply_rope_with_cos_sin_cache_inplace:
            positions = torch.arange(
                workload.sequence_length, device="cuda", dtype=torch.long
            )
            cos_sin_cache = torch.cat((cos, sin), dim=-1)
            flashinfer_input_before = hidden.clone()

            def flashinfer_preserving_inputs():
                query_output = hidden.clone()
                key_output = hidden.clone()
                apply_rope_with_cos_sin_cache_inplace(
                    positions=positions,
                    query=query_output.view(workload.sequence_length, -1),
                    key=key_output.view(workload.sequence_length, -1),
                    head_size=128,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=False,
                )
                return query_output, key_output

            minwm_key_output = torch.empty_like(hidden)

            def minwm_preserving_inputs():
                query_output = minwm_rotary(hidden, cos, sin)
                minwm_rotary_out(hidden, cos, sin, minwm_key_output)
                return query_output, minwm_key_output

            flashinfer_query, flashinfer_key = flashinfer_preserving_inputs()
            expected = torch_reference(hidden, cos, sin)
            torch.testing.assert_close(flashinfer_query, expected, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(flashinfer_key, expected, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(hidden, flashinfer_input_before, rtol=0, atol=0)
            mismatch_count = int(torch.count_nonzero(flashinfer_query != expected))
            max_abs_error = float(
                (flashinfer_query.float() - expected.float()).abs().max()
            )
            del flashinfer_query, flashinfer_key, expected, flashinfer_input_before
            torch.cuda.synchronize()

            qk_fns = {
                "minwm": minwm_preserving_inputs,
                "flashinfer": flashinfer_preserving_inputs,
            }
            qk_order = list(qk_fns)
            random.shuffle(qk_order)
            qk_times = {name: cuda_event_us(qk_fns[name]) for name in qk_order}
            qk_peak_memory = {name: peak_extra_mib(qk_fns[name]) for name in qk_order}
            flashinfer_results.append(
                (
                    workload.name,
                    f"[1,{workload.sequence_length},{workload.num_heads},128]",
                    qk_times,
                    qk_peak_memory,
                    mismatch_count,
                    max_abs_error,
                )
            )
        torch.cuda.empty_cache()

    print()
    print(
        "| workload | shape | rotate+copy GPU us | direct-out GPU us | speedup | "
        "rotate+copy host us | direct-out host us | rotate+copy peak MiB | "
        "direct-out peak MiB |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, shape, times, host_times, peak_memory in cache_store_results:
        print(
            f"| {name} | `{shape}` | {times['copy']:.2f} | {times['out']:.2f} | "
            f"{times['copy'] / times['out']:.2f}x | {host_times['copy']:.2f} | "
            f"{host_times['out']:.2f} | {peak_memory['copy']:.2f} | "
            f"{peak_memory['out']:.2f} |"
        )

    print()
    if not apply_rope_with_cos_sin_cache_inplace:
        print("FlashInfer unavailable; LingBot-compatible Q+K comparison skipped.")
        return
    print(
        "| workload | shape | MinWM exact Q+direct-K us | FlashInfer preserve-input "
        "Q+K us | MinWM speedup | MinWM peak MiB | FlashInfer peak MiB | "
        "FlashInfer bitwise mismatches | FlashInfer max abs error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (
        name,
        shape,
        times,
        peak_memory,
        mismatch_count,
        max_abs_error,
    ) in flashinfer_results:
        print(
            f"| {name} | `{shape}` | {times['minwm']:.2f} | "
            f"{times['flashinfer']:.2f} | "
            f"{times['flashinfer'] / times['minwm']:.2f}x | "
            f"{peak_memory['minwm']:.2f} | {peak_memory['flashinfer']:.2f} | "
            f"{mismatch_count} | {max_abs_error:.8f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="also compare static fullgraph Inductor against the Hopper kernel",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead"),
        default="default",
    )
    args = parser.parse_args()
    benchmark()
    if args.torch_compile:
        benchmark_torch_compile(
            None if args.torch_compile_mode == "default" else args.torch_compile_mode
        )
    sys.exit(0)
