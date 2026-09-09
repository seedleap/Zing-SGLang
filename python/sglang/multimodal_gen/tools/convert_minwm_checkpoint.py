# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0
"""Convert the public Zing-0.5 release into an SGLang model directory.

The preferred input is the directory downloaded from ``seedleap/Zing-0.5``.
Legacy explicit checkpoint/component arguments remain available for compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file

GENERATOR_KEYS = ("generator_ema", "ema_student", "generator", "model")
WRAPPER_PREFIXES = (
    "model._fsdp_wrapped_module.",
    "_fsdp_wrapped_module.",
    "module.",
    "model.",
)
DONOR_COMPONENTS = ("text_encoder", "tokenizer", "vae")
DEFAULT_SOURCE_URI = "modelscope://seedleap/Zing-0.5"

ACTION_HIDDEN_BIAS_KEYS = (
    "action_in.fuse.0.bias",
    "action_in.fuse.2.bias",
    "action_in.encode_1.conv.bias",
    "action_in.encode_1.norm.bias",
    "action_in.encode_2.conv.bias",
    "action_in.encode_2.norm.bias",
)

TRANSFORMER_CONFIG = {
    "_class_name": "MinWMCausalTransformer3DModel",
    "_diffusers_version": "0.36.0",
    "model_type": "t2v",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "in_dim": 48,
    "out_dim": 48,
    "dim": 3072,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "in_channels": 48,
    "out_channels": 48,
    "num_heads": 24,
    "num_layers": 30,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "qk_norm": "rms_norm_across_heads",
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
    "local_attn_size": 97,
    "sink_size": 9,
    "rope_position_mode": "absolute",
    "rope_max_frame_gap": 1,
    "prompt_first_frame_pin_enabled": False,
    "scene_cut_rope_offset": 0,
    "scene_cut_sink_enabled": False,
    "num_frame_per_block": 4,
    "num_frame_first_block": 1,
    "num_frames_per_block": 4,
    "sliding_window_num_frames": 97,
    "action_type": "primitive_token_residual",
    "action_embed_dim": 256,
    "action_hidden_dim": 512,
    "action_kernel_size": 3,
    "action_history_frames": 4,
    "action_non_proj_bias": True,
}

MODEL_INDEX = {
    "_class_name": "ZingCausalDMDPipeline",
    "_diffusers_version": "0.36.0",
    # The runtime constructs the four-step DMD scheduler. No scheduler files are
    # present in the public Zing release or required in the converted artifact.
    "scheduler": None,
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "MinWMCausalTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def _select_generator_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Zing checkpoint must contain a state dict")
    for key in GENERATOR_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key
    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint, "root"
    raise ValueError(
        f"no generator state dict found; checked wrapper keys {GENERATOR_KEYS}"
    )


def _strip_wrapper_prefix(name: str) -> str:
    previous = None
    while name != previous:
        previous = name
        for prefix in WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    return name


def extract_generator_state_dict(checkpoint_path: str, *, mmap: bool = True):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=mmap,
    )
    state_dict, selected_key = _select_generator_state_dict(checkpoint)
    cleaned = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"non-tensor checkpoint entry: {name}")
        # Keep mmap-backed storages lazy. Individual shards are made contiguous
        # immediately before writing so peak memory is bounded by shard size.
        cleaned[_strip_wrapper_prefix(name)] = tensor.detach()
    return cleaned, selected_key


def validate_generator_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    block_indices = {
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("blocks.") and name.split(".")[1].isdigit()
    }
    required_shapes = {
        "patch_embedding.weight": (3072, 48, 1, 2, 2),
        "patch_embedding.bias": (3072,),
        "action_in.encode_1.conv.weight": (512, 512, 3),
        "action_in.encode_2.conv.weight": (512, 512, 3),
        "action_in.proj.weight": (3072, 512),
        "action_in.proj.bias": (3072,),
        "head.head.weight": (192, 3072),
    }
    has_move_embedding = "action_in.move_embedding.weight" in state_dict
    has_look_embedding = "action_in.look_embedding.weight" in state_dict
    if has_move_embedding != has_look_embedding:
        raise ValueError("incompatible Zing checkpoint: incomplete action embeddings")
    action_type = (
        "primitive_token_residual"
        if has_move_embedding
        else "primitive_rope_token_residual"
    )
    present_hidden_biases = {
        name for name in ACTION_HIDDEN_BIAS_KEYS if name in state_dict
    }
    if action_type == "primitive_rope_token_residual":
        if present_hidden_biases and len(present_hidden_biases) != len(
            ACTION_HIDDEN_BIAS_KEYS
        ):
            missing = sorted(set(ACTION_HIDDEN_BIAS_KEYS) - present_hidden_biases)
            errors = [
                "partial primitive RoPE action hidden biases; missing "
                + ", ".join(missing)
            ]
        else:
            errors = []
        action_non_proj_bias = bool(present_hidden_biases)
    else:
        missing = sorted(set(ACTION_HIDDEN_BIAS_KEYS) - present_hidden_biases)
        errors = (
            ["primitive token action checkpoint is missing " + ", ".join(missing)]
            if missing
            else []
        )
        action_non_proj_bias = True
    if present_hidden_biases:
        required_shapes.update({name: (512,) for name in ACTION_HIDDEN_BIAS_KEYS})
    if has_move_embedding:
        required_shapes.update(
            {
                "action_in.move_embedding.weight": (5, 256),
                "action_in.look_embedding.weight": (5, 256),
            }
        )
    for name, shape in required_shapes.items():
        tensor = state_dict.get(name)
        if tensor is None:
            errors.append(f"missing {name}")
        elif tuple(tensor.shape) != shape:
            errors.append(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
    if block_indices != set(range(30)):
        errors.append(f"expected transformer blocks 0..29, got {sorted(block_indices)}")
    forbidden = [
        name
        for name in state_dict
        if "prope" in name.lower() or "camera" in name.lower()
    ]
    if forbidden:
        errors.append(
            "checkpoint contains old camera/PRoPE tensors: " + ", ".join(forbidden[:8])
        )
    if errors:
        raise ValueError("incompatible Zing checkpoint:\n- " + "\n- ".join(errors))
    return {
        "tensor_count": len(state_dict),
        "parameter_count": sum(tensor.numel() for tensor in state_dict.values()),
        "block_count": len(block_indices),
        "action_type": action_type,
        "action_non_proj_bias": action_non_proj_bias,
    }


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor], output_dir: Path, max_shard_bytes: int
) -> dict:
    shards: list[OrderedDict[str, torch.Tensor]] = []
    current: OrderedDict[str, torch.Tensor] = OrderedDict()
    current_bytes = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        tensor_bytes = _tensor_nbytes(tensor)
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            shards.append(current)
            current = OrderedDict()
            current_bytes = 0
        current[name] = tensor
        current_bytes += tensor_bytes
    if current:
        shards.append(current)

    weight_map = {}
    total = len(shards)
    for index, shard in enumerate(shards, start=1):
        filename = f"diffusion_pytorch_model-{index:05d}-of-{total:05d}.safetensors"
        contiguous_shard = OrderedDict(
            (name, tensor.contiguous()) for name, tensor in shard.items()
        )
        save_file(
            contiguous_shard,
            output_dir / filename,
            metadata={"format": "pt"},
        )
        del contiguous_shard
        for name in shard:
            weight_map[name] = filename
    index = {
        "metadata": {"total_size": sum(_tensor_nbytes(t) for t in state_dict.values())},
        "weight_map": weight_map,
    }
    with (output_dir / "diffusion_pytorch_model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"shard_count": total, **index["metadata"]}


def link_or_copy(source: Path, target: Path, *, link: bool) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing component: {target}")
    if link:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def build_transformer_config(
    *,
    local_attn_size: int,
    sink_size: int,
    sliding_window_num_frames: int,
    action_type: str = "primitive_token_residual",
    action_non_proj_bias: bool = True,
    rope_position_mode: str = "absolute",
    rope_max_frame_gap: int = 1,
    prompt_first_frame_pin_enabled: bool = False,
    scene_cut_rope_offset: int = 0,
    scene_cut_sink_enabled: bool = False,
) -> dict:
    if local_attn_size != -1 and local_attn_size <= 0:
        raise ValueError("local_attn_size must be -1 or positive")
    if sink_size < 0:
        raise ValueError("sink_size must be non-negative")
    if local_attn_size != -1 and sink_size >= local_attn_size:
        raise ValueError("sink_size must be smaller than finite local_attn_size")
    if sliding_window_num_frames <= 0:
        raise ValueError("sliding_window_num_frames must be positive")
    if local_attn_size != -1 and sliding_window_num_frames != local_attn_size:
        raise ValueError(
            "bounded Zing requires sliding_window_num_frames to equal local_attn_size"
        )
    if local_attn_size == -1 and sink_size >= sliding_window_num_frames:
        raise ValueError("sink_size must be smaller than sliding_window_num_frames")
    if action_type not in {
        "primitive_token_residual",
        "primitive_rope_token_residual",
    }:
        raise ValueError(f"unsupported Zing action_type={action_type!r}")
    if rope_position_mode not in {"absolute", "block_relative"}:
        raise ValueError("rope_position_mode must be absolute or block_relative")
    if rope_max_frame_gap < 1:
        raise ValueError("rope_max_frame_gap must be >= 1")
    if rope_position_mode == "block_relative" and scene_cut_rope_offset:
        raise ValueError(
            "block_relative RoPE does not support nonzero scene_cut_rope_offset"
        )
    config = dict(TRANSFORMER_CONFIG)
    config.update(
        {
            "local_attn_size": local_attn_size,
            "sink_size": sink_size,
            "sliding_window_num_frames": sliding_window_num_frames,
            "action_type": action_type,
            "action_non_proj_bias": action_non_proj_bias,
            "rope_position_mode": rope_position_mode,
            "rope_max_frame_gap": rope_max_frame_gap,
            "prompt_first_frame_pin_enabled": prompt_first_frame_pin_enabled,
            "scene_cut_rope_offset": scene_cut_rope_offset,
            "scene_cut_sink_enabled": scene_cut_sink_enabled,
        }
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zing-dir",
        help=(
            "Directory downloaded from seedleap/Zing-0.5, containing "
            "generator/model.pt and pretrained/"
        ),
    )
    parser.add_argument(
        "--minwm-checkpoint",
        help="Legacy explicit path to generator/model.pt",
    )
    parser.add_argument(
        "--donor-diffusers-dir",
        help="Legacy explicit directory containing text_encoder, tokenizer, and vae",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--link-donor", action="store_true")
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE_URI)
    parser.add_argument("--source-version-id")
    parser.add_argument("--source-etag")
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    parser.add_argument(
        "--no-mmap",
        action="store_false",
        dest="mmap",
        help="Disable memory-mapped checkpoint loading",
    )
    parser.add_argument("--local-attn-size", type=int, default=97)
    parser.add_argument("--sink-size", type=int, default=9)
    parser.add_argument(
        "--sliding-window-num-frames",
        type=int,
        help="Defaults to --local-attn-size for a bounded cache",
    )
    parser.add_argument(
        "--action-type",
        choices=(
            "auto",
            "primitive_token_residual",
            "primitive_rope_token_residual",
        ),
        default="auto",
    )
    parser.add_argument(
        "--rope-position-mode",
        choices=("absolute", "block_relative"),
        default="absolute",
    )
    parser.add_argument("--rope-max-frame-gap", type=int, default=1)
    parser.add_argument("--prompt-first-frame-pin-enabled", action="store_true")
    parser.add_argument("--scene-cut-rope-offset", type=int, default=0)
    parser.add_argument("--scene-cut-sink-enabled", action="store_true")
    return parser.parse_args()


def resolve_source_layout(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve either the public Zing layout or the legacy explicit layout."""

    if args.zing_dir:
        if args.minwm_checkpoint or args.donor_diffusers_dir:
            raise ValueError(
                "--zing-dir cannot be combined with --minwm-checkpoint or "
                "--donor-diffusers-dir"
            )
        zing_dir = Path(args.zing_dir)
        checkpoint = zing_dir / "generator" / "model.pt"
        components = zing_dir / "pretrained"
    else:
        if not args.minwm_checkpoint or not args.donor_diffusers_dir:
            raise ValueError(
                "provide --zing-dir, or provide both --minwm-checkpoint and "
                "--donor-diffusers-dir"
            )
        checkpoint = Path(args.minwm_checkpoint)
        components = Path(args.donor_diffusers_dir)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Zing generator checkpoint: {checkpoint}")
    if not components.is_dir():
        raise FileNotFoundError(f"missing Zing pretrained directory: {components}")
    return checkpoint, components


def main() -> None:
    args = parse_args()
    if args.max_shard_gib <= 0:
        raise ValueError("--max-shard-gib must be positive")
    checkpoint, donor = resolve_source_layout(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict, selected_key = extract_generator_state_dict(
        str(checkpoint), mmap=args.mmap
    )
    summary = validate_generator_state_dict(state_dict)
    action_type = (
        summary["action_type"] if args.action_type == "auto" else args.action_type
    )
    if action_type != summary["action_type"]:
        raise ValueError(
            f"--action-type={action_type} does not match checkpoint tensors "
            f"({summary['action_type']})"
        )
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir()
    shard_summary = save_sharded_state_dict(
        state_dict,
        transformer_dir,
        max_shard_bytes=int(args.max_shard_gib * 1024**3),
    )
    transformer_config = build_transformer_config(
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        sliding_window_num_frames=(
            args.sliding_window_num_frames
            if args.sliding_window_num_frames is not None
            else args.local_attn_size
        ),
        action_type=action_type,
        action_non_proj_bias=summary["action_non_proj_bias"],
        rope_position_mode=args.rope_position_mode,
        rope_max_frame_gap=args.rope_max_frame_gap,
        prompt_first_frame_pin_enabled=args.prompt_first_frame_pin_enabled,
        scene_cut_rope_offset=args.scene_cut_rope_offset,
        scene_cut_sink_enabled=args.scene_cut_sink_enabled,
    )
    with (transformer_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(transformer_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for component in DONOR_COMPONENTS:
        source = donor / component
        if not source.exists():
            raise FileNotFoundError(f"missing Zing component: {source}")
        link_or_copy(source, output_dir / component, link=args.link_donor)
    with (output_dir / "model_index.json").open("w", encoding="utf-8") as handle:
        json.dump(MODEL_INDEX, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "format": "sglang-zing-0.5-v1",
        "source_checkpoint": {
            "uri": args.source_uri,
            "version_id": args.source_version_id,
            "etag": args.source_etag,
            "local_size": os.path.getsize(checkpoint),
            "selected_state_dict": selected_key,
        },
        "components": list(DONOR_COMPONENTS),
        "generator": summary,
        "safetensors": shard_summary,
        "native_geometry": {"width": 832, "height": 480},
        "causal_cache_defaults": {
            "local_attn_size": transformer_config["local_attn_size"],
            "sink_size": transformer_config["sink_size"],
            "sliding_window_num_frames": transformer_config[
                "sliding_window_num_frames"
            ],
            "provenance": "converter arguments, not checkpoint tensors",
        },
    }
    with (output_dir / "zing_conversion_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        "serve with: python -m sglang.multimodal_gen.runtime.launch_server "
        f"--model-path {output_dir} --pipeline-class-name ZingCausalDMDPipeline"
    )


if __name__ == "__main__":
    main()
