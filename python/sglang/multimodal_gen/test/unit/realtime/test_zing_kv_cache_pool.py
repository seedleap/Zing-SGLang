# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.models.dits.zing_kv_cache import (
    MinWMCausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing import (
    zing_causal_denoising as zing_stage_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing.zing_causal_denoising import (
    MinWMCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.realtime.causal_kv_cache_pool import (
    RealtimeKVCachePool,
    RealtimeKVCachePoolCapacityError,
    RealtimeKVCachePoolSlot,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.runtime.realtime.states import (
    get_realtime_causal_dit_state,
)


def _cache_slot(slot_id: int) -> RealtimeKVCachePoolSlot:
    kv_cache = MinWMCausalSelfAttentionKVCache(
        k=torch.empty(1, 8, 2, 4),
        v=torch.empty(1, 8, 2, 4),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
    )
    crossattn_cache = CrossAttentionKVCache(
        k=torch.empty(1, 4, 2, 4),
        v=torch.empty(1, 4, 2, 4),
    )
    return RealtimeKVCachePoolSlot(slot_id, [kv_cache], [crossattn_cache])


def test_realtime_kv_cache_pool_reuses_slot_and_state_dispose_releases_it():
    slot = _cache_slot(0)
    pool = RealtimeKVCachePool([slot])

    first = pool.acquire(bucket_key="832x480")
    assert first.kv_cache is slot.kv_cache
    assert first.bucket_key == "832x480"
    with pytest.raises(RealtimeKVCachePoolCapacityError):
        pool.acquire()

    first.release()
    first.release()
    assert pool.available == 1

    second = pool.acquire()
    state = get_realtime_causal_dit_state(RealtimeSession())
    state.kv_cache_pool_lease = second
    state.kv_cache = second.kv_cache
    state.crossattn_cache = second.crossattn_cache
    state.dispose()

    assert pool.available == 1
    assert pool.acquire_count == 2
    assert pool.release_count == 2


def test_realtime_kv_cache_pool_honors_capacity_and_reuses_slots_fifo():
    first_slot = _cache_slot(0)
    second_slot = _cache_slot(1)
    pool = RealtimeKVCachePool([first_slot, second_slot])

    first = pool.acquire()
    second = pool.acquire()
    with pytest.raises(RealtimeKVCachePoolCapacityError):
        pool.acquire()

    first.release()
    second.release()

    assert pool.acquire().slot_id == first_slot.slot_id
    assert pool.acquire().slot_id == second_slot.slot_id


def test_minwm_pool_rejects_capacity_below_worker_admission_limit():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_kv_cache_pool_size=1,
            realtime_causal_kv_cache_pool_buckets="480p",
        ),
        realtime_max_sessions_per_worker=2,
    )

    with pytest.raises(ValueError, match="must cover every admitted worker session"):
        stage.initialize_realtime_kv_cache_pool(server_args)


def test_minwm_pool_defaults_off_when_no_launch_contract_is_configured():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage._realtime_kv_cache_pool = None
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(),
        realtime_max_sessions_per_worker=1,
    )

    stage.initialize_realtime_kv_cache_pool(server_args)

    assert stage.realtime_kv_cache_pool_enabled is False


@pytest.mark.parametrize(
    ("pipeline_config", "message"),
    (
        (
            SimpleNamespace(realtime_causal_kv_cache_pool_buckets="480p"),
            "requires realtime_causal_kv_cache_pool_size",
        ),
        (
            SimpleNamespace(realtime_causal_kv_cache_pool_size=1),
            "requires symbolic resolution buckets",
        ),
    ),
)
def test_minwm_pool_rejects_partial_launch_contract(pipeline_config, message):
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    server_args = SimpleNamespace(
        pipeline_config=pipeline_config,
        realtime_max_sessions_per_worker=1,
    )

    with pytest.raises(ValueError, match=message):
        stage.initialize_realtime_kv_cache_pool(server_args)


def test_minwm_pool_is_fully_allocated_during_stage_initialization(monkeypatch):
    arch_config = SimpleNamespace(
        patch_size=(1, 2, 2),
        sink_size=0,
        sliding_window_num_frames=128,
        rope_position_mode="absolute",
        rope_max_frame_gap=1,
        prompt_first_frame_pin_enabled=False,
        scene_cut_rope_offset=0,
        scene_cut_sink_enabled=False,
    )
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=arch_config,
        num_attention_heads=4,
        attention_head_dim=8,
    )
    stage.local_attn_size = -1
    stage.sink_size = 0
    stage.sliding_window_num_frames = 128
    stage.num_transformer_blocks = 2
    stage.num_frames_per_block = 4
    stage.causal_kv_cache = None
    stage.crossattn_cache = None
    stage._minwm_cuda_graph_enabled = False
    stage._minwm_cuda_graph_runner = None
    stage._realtime_kv_cache_pool = None
    stage._realtime_kv_cache_pool_policies = {}
    stage._realtime_kv_cache_pool_shapes = {}
    stage._realtime_kv_cache_pool_bucket_by_shape = {}
    stage._realtime_kv_cache_pool_dtype = None

    pipeline_config = SimpleNamespace(
        realtime_causal_kv_cache_pool_size=2,
        realtime_causal_kv_cache_pool_buckets="480p,720p,720p-wide",
        realtime_causal_kv_cache_num_frames=32,
        realtime_causal_sink_size=8,
        vae_config=SimpleNamespace(
            arch_config=SimpleNamespace(scale_factor_spatial=16)
        ),
        text_encoder_configs=[SimpleNamespace(arch_config=SimpleNamespace(text_len=4))],
    )
    server_args = SimpleNamespace(
        pipeline_config=pipeline_config,
        realtime_max_sessions_per_worker=2,
        enable_cuda_graph=False,
    )
    monkeypatch.setattr(
        zing_stage_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(zing_stage_module, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(zing_stage_module, "get_sp_parallel_rank", lambda: 0)
    allocations = []

    def fake_initialize_causal_caches(**kwargs):
        allocations.append(kwargs)
        slot = _cache_slot(len(allocations) - 1)
        return slot.kv_cache, slot.crossattn_cache

    monkeypatch.setattr(
        stage, "_initialize_causal_caches", fake_initialize_causal_caches
    )

    stage.initialize_realtime_kv_cache_pool(server_args)

    assert stage.realtime_kv_cache_pool_enabled is True
    assert stage._realtime_kv_cache_pool.size == 2
    assert stage._realtime_kv_cache_pool.available == 2
    assert stage._realtime_kv_cache_pool_shapes == {
        "832x480": (1, 30, 52),
        "1248x704": (1, 44, 78),
        "1280x704": (1, 44, 80),
    }
    assert (
        stage._realtime_kv_cache_pool_policies["832x480"].expected_cache_tokens == 12480
    )
    assert (
        stage._realtime_kv_cache_pool_policies["1248x704"].expected_cache_tokens
        == 27456
    )
    assert (
        stage._realtime_kv_cache_pool_policies["1280x704"].expected_cache_tokens
        == 28160
    )
    assert all(
        item["kv_cache_kwargs"]["kv_cache_size"] == 28160 for item in allocations
    )
    assert all(
        cache.rotated_k is not None
        for slot in stage._realtime_kv_cache_pool._slots
        for cache in slot.kv_cache
    )
    assert len(allocations) == 2
    assert all(item["batch_size"] == 1 for item in allocations)


def test_minwm_realtime_requests_reset_and_reuse_startup_pool_slot(monkeypatch):
    slot = _cache_slot(0)
    pool = RealtimeKVCachePool([slot])
    policy = CausalDMDCachePolicy(
        sequence_shard_enabled=False,
        num_attention_heads=2,
        expected_cache_tokens=8,
        expected_sink_tokens=0,
        kv_cache_kwargs={"allow_growth": False},
    )
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage._realtime_kv_cache_pool = pool
    stage._realtime_kv_cache_pool_policies = {"832x480": policy}
    stage._realtime_kv_cache_pool_shapes = {"832x480": (1, 2, 3)}
    stage._realtime_kv_cache_pool_bucket_by_shape = {(1, 2, 3): "832x480"}
    stage._realtime_kv_cache_pool_dtype = torch.bfloat16
    stage._current_use_nvtx = False
    monkeypatch.setattr(
        stage, "_build_realtime_causal_cache_policy", lambda _batch, _args: policy
    )
    resets = []
    monkeypatch.setattr(
        stage,
        "_reset_causal_caches",
        lambda **kwargs: resets.append(kwargs),
    )
    monkeypatch.setattr(
        stage, "_log_runtime_alignment_once", lambda _batch, _cache_ctx: None
    )
    ctx = SimpleNamespace(
        batch_size=1,
        height=2,
        width=3,
        target_dtype=torch.bfloat16,
    )
    session = RealtimeSession()
    first_batch = SimpleNamespace(
        session=session,
        block_idx=0,
        condition_inputs={},
        image_latent=None,
    )
    next_batch = SimpleNamespace(
        session=session,
        block_idx=1,
        condition_inputs={},
        image_latent=None,
    )

    first = stage._prepare_realtime_causal_caches(first_batch, SimpleNamespace(), ctx)
    second = stage._prepare_realtime_causal_caches(next_batch, SimpleNamespace(), ctx)

    assert first.kv_cache is slot.kv_cache
    assert second.kv_cache is first.kv_cache
    assert pool.acquire_count == 1
    assert len(resets) == 1

    session.dispose()
    replacement = stage._prepare_realtime_causal_caches(
        SimpleNamespace(
            session=RealtimeSession(),
            block_idx=0,
            condition_inputs={},
            image_latent=None,
        ),
        SimpleNamespace(),
        ctx,
    )
    assert replacement.kv_cache is first.kv_cache
    assert pool.acquire_count == 2
    assert len(resets) == 2


def test_minwm_pool_shape_mismatch_fails_before_acquiring_slot(monkeypatch):
    slot = _cache_slot(0)
    pool = RealtimeKVCachePool([slot])
    policy = CausalDMDCachePolicy(
        sequence_shard_enabled=False,
        num_attention_heads=2,
        expected_cache_tokens=8,
        expected_sink_tokens=0,
        kv_cache_kwargs={"allow_growth": False},
    )
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage._realtime_kv_cache_pool = pool
    stage._realtime_kv_cache_pool_policies = {"832x480": policy}
    stage._realtime_kv_cache_pool_shapes = {"832x480": (1, 2, 3)}
    stage._realtime_kv_cache_pool_bucket_by_shape = {(1, 2, 3): "832x480"}
    stage._realtime_kv_cache_pool_dtype = torch.bfloat16
    monkeypatch.setattr(
        stage, "_build_realtime_causal_cache_policy", lambda _batch, _args: policy
    )

    with pytest.raises(ValueError, match="latent_shape"):
        stage._prepare_realtime_causal_caches_from_startup_pool(
            SimpleNamespace(session=RealtimeSession(), block_idx=0),
            SimpleNamespace(),
            SimpleNamespace(
                batch_size=1,
                height=3,
                width=3,
                target_dtype=torch.bfloat16,
            ),
        )

    assert pool.available == 1
    assert pool.acquire_count == 0


def test_minwm_pool_reuses_max_backing_for_all_resolution_buckets(monkeypatch):
    slot = _cache_slot(0)
    pool = RealtimeKVCachePool([slot])
    policies = {
        "832x480": CausalDMDCachePolicy(
            sequence_shard_enabled=False,
            num_attention_heads=2,
            expected_cache_tokens=4,
            expected_sink_tokens=1,
            kv_cache_kwargs={"allow_growth": False},
        ),
        "1248x704": CausalDMDCachePolicy(
            sequence_shard_enabled=False,
            num_attention_heads=2,
            expected_cache_tokens=6,
            expected_sink_tokens=2,
            kv_cache_kwargs={"allow_growth": False},
        ),
        "1280x704": CausalDMDCachePolicy(
            sequence_shard_enabled=False,
            num_attention_heads=2,
            expected_cache_tokens=8,
            expected_sink_tokens=3,
            kv_cache_kwargs={"allow_growth": False},
        ),
    }
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage._realtime_kv_cache_pool = pool
    stage._realtime_kv_cache_pool_policies = policies
    stage._realtime_kv_cache_pool_shapes = {
        "832x480": (1, 2, 3),
        "1248x704": (1, 3, 4),
        "1280x704": (1, 3, 5),
    }
    stage._realtime_kv_cache_pool_bucket_by_shape = {
        (1, 2, 3): "832x480",
        (1, 3, 4): "1248x704",
        (1, 3, 5): "1280x704",
    }
    stage._realtime_kv_cache_pool_dtype = torch.bfloat16
    stage._current_use_nvtx = False
    monkeypatch.setattr(
        stage,
        "_build_realtime_causal_cache_policy",
        lambda batch, _args: policies[batch.bucket],
    )
    monkeypatch.setattr(stage, "_reset_causal_caches", lambda **_kwargs: None)

    session = RealtimeSession()
    batch_480p = SimpleNamespace(session=session, block_idx=0, bucket="832x480")
    ctx_480p = SimpleNamespace(
        batch_size=1,
        height=2,
        width=3,
        target_dtype=torch.bfloat16,
    )
    stage._prepare_realtime_causal_caches_from_startup_pool(
        batch_480p, SimpleNamespace(), ctx_480p
    )

    cache = slot.kv_cache[0]
    assert cache.k.shape[1] == 8
    assert cache.cache_size == 4
    assert cache.attention_window_size == 4
    assert cache.sink_tokens == 1

    with pytest.raises(ValueError, match="cannot change within a realtime session"):
        stage._prepare_realtime_causal_caches_from_startup_pool(
            SimpleNamespace(session=session, block_idx=1, bucket="1280x704"),
            SimpleNamespace(),
            SimpleNamespace(
                batch_size=1,
                height=3,
                width=5,
                target_dtype=torch.bfloat16,
            ),
        )

    session.dispose()
    session_1248 = RealtimeSession()
    stage._prepare_realtime_causal_caches_from_startup_pool(
        SimpleNamespace(
            session=session_1248,
            block_idx=0,
            bucket="1248x704",
        ),
        SimpleNamespace(),
        SimpleNamespace(
            batch_size=1,
            height=3,
            width=4,
            target_dtype=torch.bfloat16,
        ),
    )
    assert cache.k.shape[1] == 8
    assert cache.cache_size == 6
    assert cache.attention_window_size == 6
    assert cache.sink_tokens == 2

    session_1248.dispose()
    stage._prepare_realtime_causal_caches_from_startup_pool(
        SimpleNamespace(
            session=RealtimeSession(),
            block_idx=0,
            bucket="1280x704",
        ),
        SimpleNamespace(),
        SimpleNamespace(
            batch_size=1,
            height=3,
            width=5,
            target_dtype=torch.bfloat16,
        ),
    )
    assert cache.k.shape[1] == 8
    assert cache.cache_size == 8
    assert cache.attention_window_size == 8
    assert cache.sink_tokens == 3
