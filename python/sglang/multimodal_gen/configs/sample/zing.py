# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam
from sglang.multimodal_gen.configs.zing_resolution_buckets import (
    MINWM_RESOLUTION_BUCKETS,
    normalize_minwm_resolution_buckets,
)


@dataclass
class MinWMSamplingParams(Wan2_2_TI2V_5B_SamplingParam):
    height: int = 480
    width: int = 832
    num_frames: int = 1
    fps: int = 24
    guidance_scale: float = 0.0
    num_inference_steps: int = 4
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(832, 480), (480, 832)]
    )

    def _adjust(self, server_args):
        configured_buckets = getattr(
            getattr(server_args, "pipeline_config", None),
            "realtime_causal_kv_cache_pool_buckets",
            None,
        )
        if configured_buckets is not None:
            bucket_names = normalize_minwm_resolution_buckets(str(configured_buckets))
            self.supported_resolutions = [
                MINWM_RESOLUTION_BUCKETS[name] for name in bucket_names
            ]
            if (
                self.width is not None
                and self.height is not None
                and (self.width, self.height) not in self.supported_resolutions
            ):
                supported = ", ".join(
                    f"{width}x{height}" for width, height in self.supported_resolutions
                )
                raise ValueError(
                    "MinWM request resolution is not enabled by the launch "
                    f"profile: {self.width}x{self.height}; supported={supported}"
                )

        enable_sequence_shard = self.enable_sequence_shard
        sp_degree = getattr(server_args, "sp_degree", 1) or 1
        if sp_degree > 1 and enable_sequence_shard is False:
            raise ValueError(
                "MinWM with sp_degree > 1 requires enable_sequence_shard=True."
            )
        if enable_sequence_shard is None or enable_sequence_shard:
            self.adjust_frames = False
        super()._adjust(server_args)
        if enable_sequence_shard is None or enable_sequence_shard:
            self.enable_sequence_shard = True
            self.adjust_frames = False
