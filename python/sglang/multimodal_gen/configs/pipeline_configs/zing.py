# Copyright 2026 Seedleap.ai
# Adapted from the Apache-2.0 Zing and Wan implementations.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for the Zing-0.5 Wan2.2-5B causal DMD model."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import ZingVideoConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig, T5Config
from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config

ZING_ACTION_LABELS_CONDITION = "zing_action_labels"
ZING_ACTION_WEIGHTS_CONDITION = "zing_action_weights"
ZING_CHUNK_SEED_CONDITION = "zing_chunk_seed"
ZING_CHUNK_SEED_PREFIX_FRAMES_CONDITION = "zing_chunk_seed_prefix_frames"
ZING_CHUNK_SEEDS_INPUT = "zing_chunk_seeds"
ZING_CONDITION_SWITCH_CONDITION = "zing_condition_switch"
ZING_PROMPT_SCHEDULE_INPUT = "zing_prompt_schedule"
ZING_PROMPT_UPDATED_CONDITION = "zing_prompt_updated"
ZING_TOTAL_CHUNKS_CONDITION = "zing_total_chunks"
ZING_TOTAL_LATENT_FRAMES_CONDITION = "zing_total_latent_frames"
ZING_VAE_LANES = ("parity", "parallel")


def _zing_native_component_names() -> tuple[str, ...]:
    vae_lane = os.environ.get("ZING_VAE_LANE")
    if vae_lane is not None:
        if vae_lane not in ZING_VAE_LANES:
            raise ValueError(
                f"ZING_VAE_LANE must be one of {ZING_VAE_LANES}, got {vae_lane!r}"
            )
        # The explicit lane takes precedence over the component list so
        # benchmark manifests cannot accidentally mix parity and speed modes.
        return ("text_encoder", "vae") if vae_lane == "parity" else ("text_encoder",)

    value = os.environ.get("ZING_NATIVE_COMPONENTS", "text_encoder,vae")
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = set(names) - {"text_encoder", "vae"}
    if unknown:
        raise ValueError(f"unknown ZING_NATIVE_COMPONENTS entries: {sorted(unknown)}")
    return names


def zing_t5_postprocess_text(outputs, _text_inputs) -> TextConditioningOutput:
    attention_mask = getattr(outputs, "attention_mask", None)
    if attention_mask is None:
        if _text_inputs is None or "attention_mask" not in _text_inputs:
            raise ValueError("Zing text encoding requires an attention mask")
        attention_mask = _text_inputs["attention_mask"]
    token_mask = attention_mask.to(dtype=torch.bool)
    hidden_state = outputs.last_hidden_state.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    # Current Zing main keeps at least 512 positions in the packed text
    # context. Positions after the true token length are explicit zero vectors,
    # but they still participate as K/V entries in cross attention. This is a
    # model contract, not ordinary attention-mask trimming.
    seq_lens = token_mask.sum(dim=1).long().clamp_min(512)
    positions = torch.arange(hidden_state.shape[1], device=hidden_state.device)
    context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    if torch.isnan(hidden_state).any():
        raise ValueError("Zing text encoder produced NaN embeddings")
    return TextConditioningOutput(
        prompt_embeds=hidden_state,
        prompt_embeds_mask=context_mask,
        prompt_seq_lens=[int(length) for length in seq_lens.tolist()],
    )


def _zing_t5_config() -> T5Config:
    return T5Config(arch_config=T5ArchConfig(text_len=1024))


@dataclass
class ZingWan22VAEArchConfig(WanVAEArchConfig):
    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    in_channels: int = 12
    out_channels: int = 12
    patch_size: int | None = 2
    is_residual: bool = True
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    latents_mean: tuple[float, ...] = (
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    )
    latents_std: tuple[float, ...] = (
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    )


@dataclass
class ZingWan22VAEConfig(WanVAEConfig):
    arch_config: ZingWan22VAEArchConfig = field(default_factory=ZingWan22VAEArchConfig)


@dataclass
class ZingCausalDMDConfig(Wan2_2_TI2V_5B_Config):
    """Exact structural/runtime defaults of the requested 5B DMD student."""

    dit_config: DiTConfig = field(default_factory=ZingVideoConfig)
    vae_config: VAEConfig = field(default_factory=ZingWan22VAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (_zing_t5_config(),)
    )
    postprocess_text_funcs: tuple = field(
        default_factory=lambda: (zing_t5_postprocess_text,)
    )
    # Zing main uses diffusers.AutoencoderKLWan and the HF UMT5 encoder
    # directly. Their tiny BF16 kernel differences are amplified by causal
    # rollout, so parity takes precedence over SGLang's optimized variants.
    native_component_names: tuple[str, ...] = field(
        default_factory=_zing_native_component_names
    )
    # Zing main executes BF16 modules directly.  An additional autocast scope
    # changes Wan VAE and DiT kernel promotion/rounding, then causal KV reuse
    # amplifies the first-step drift across chunks.
    enable_autocast: bool = False
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True
    context_noise: int = 0
    vae_precision: str = "bf16"
    preprocess_vae_encode_before_dtype_cast: bool = True
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    realtime_causal_sink_size: int | None = None
    realtime_causal_kv_cache_num_frames: int | None = None
    realtime_causal_kv_cache_pool_size: int | None = None
    realtime_causal_kv_cache_pool_buckets: str | None = None

    def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs):
        # Zing main pads to the longest prompt first, then enforces a 512-token
        # floor. Running UMT5 over a fixed 1024-token grid changes BF16 kernel
        # rounding even though the extra padding is masked out.
        tokens = tokenizer(prompt, **(tok_kwargs | {"padding": "longest"}))
        sequence_length = tokens["input_ids"].shape[1]
        if sequence_length < 512:
            padding = 512 - sequence_length
            tokens["input_ids"] = torch.nn.functional.pad(
                tokens["input_ids"],
                (0, padding),
                value=tokenizer.pad_token_id,
            )
            tokens["attention_mask"] = torch.nn.functional.pad(
                tokens["attention_mask"], (0, padding)
            )
        return tokens

    @staticmethod
    def _native_vae_stats(latents: torch.Tensor, vae):
        mean = torch.tensor(
            vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        return mean, std

    def preprocess_vae_encode(self, image, _vae):
        # The Zing processor preserves uint8 pixels until the GPU, casts them
        # to BF16, then performs div(127.5)-1 in BF16. The generic image stage
        # normalizes in FP32 on CPU first; reconstruct the lossless uint8 grid
        # so rounding and VAE inputs match main exactly.
        pixels = ((image + 1.0) * 127.5).round_().clamp_(0, 255)
        normalized = pixels.to(torch.bfloat16).div_(127.5).sub_(1.0)
        dump_root = os.environ.get("ZING_PARITY_DUMP_DIR")
        if dump_root:
            dump_dir = Path(dump_root) / "sglang"
            dump_dir.mkdir(parents=True, exist_ok=True)
            torch.save(normalized.detach().cpu(), dump_dir / "vae_input.pt")
        return normalized

    def normalize_vae_encode(self, image_latents, vae):
        mean, std = self._native_vae_stats(image_latents, vae)
        normalized = (image_latents - mean) / std
        # WanVAEWrapper returns FP32, then WanPackedProcessor serializes the
        # reference latent as FP16 before V3 moves it back to BF16.  This wire
        # boundary is numerically visible and must be reproduced in-process.
        return normalized.float().to(torch.float16).to(torch.bfloat16)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # Zing decoding multiplies by std directly. Returning identity here
        # avoids the generic algebraically-equivalent `latent / (1 / std)`,
        # whose BF16 rounding differs.
        del device, dtype, vae
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        del server_args
        if vae is None:
            raise ValueError("Zing decoding requires the native VAE")
        mean, std = self._native_vae_stats(latents, vae)
        return latents * std + mean

    def preprocess_realtime_condition_image(self, batch, _vae_image_processor) -> bool:
        if batch.condition_image is None:
            return False
        width = int(batch.width or 832)
        height = int(batch.height or 480)
        batch.condition_image = batch.condition_image.resize((width, height))
        batch.width = width
        batch.height = height
        return True

    def postprocess_image_latent(self, latent_condition, _batch):
        # PipelineConfig's generic I2V hook prepends a four-channel temporal
        # mask. Zing V3 commits the clean 48-channel VAE latent directly.
        expected_channels = self.vae_config.arch_config.z_dim
        if latent_condition.shape[1] != expected_channels:
            raise ValueError(
                "Zing reference latent must have "
                f"{expected_channels} channels, got {latent_condition.shape[1]}"
            )
        return latent_condition

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        # The condition image is encoded once per session, so sharding encode
        # does not improve steady-state FPS.  It does, however, select spatial
        # kernels with different BF16 rounding and changes the I2V condition
        # latent at SP > 1.  Keep encode replicated and spend SP only on the
        # per-chunk decode hot path.
        self.vae_config.use_parallel_encode = False
