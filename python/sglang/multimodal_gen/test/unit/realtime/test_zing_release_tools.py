# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.tools.verify_zing_artifact import validate_artifact


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_artifact(root: Path) -> Path:
    for component in ("text_encoder", "tokenizer", "vae"):
        directory = root / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}\n", encoding="utf-8")

    transformer = root / "transformer"
    transformer.mkdir()
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.float32)}, transformer / shard_name)
    _write_json(
        transformer / "config.json",
        {
            "_class_name": "MinWMCausalTransformer3DModel",
            "local_attn_size": 97,
            "sink_size": 9,
            "sliding_window_num_frames": 97,
        },
    )
    _write_json(
        transformer / "diffusion_pytorch_model.safetensors.index.json",
        {
            "metadata": {"total_size": 8},
            "weight_map": {"weight": shard_name},
        },
    )
    _write_json(
        root / "model_index.json",
        {
            "_class_name": "ZingCausalDMDPipeline",
            "scheduler": None,
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "MinWMCausalTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    )
    _write_json(
        root / "zing_conversion_manifest.json",
        {
            "format": "sglang-zing-0.5-v1",
            "source_checkpoint": {"uri": "modelscope://seedleap/Zing-0.5"},
            "components": ["text_encoder", "tokenizer", "vae"],
            "generator": {"tensor_count": 1},
            "safetensors": {"total_size": 8},
        },
    )
    return root


def test_validate_zing_artifact(tmp_path: Path):
    summary = validate_artifact(_build_artifact(tmp_path))

    assert summary["pipeline"] == "ZingCausalDMDPipeline"
    assert summary["shard_count"] == 1
    assert summary["tensor_count"] == 1
    assert summary["cache"] == {"window": 97, "sink": 9}


def test_validate_zing_artifact_rejects_private_metadata(tmp_path: Path):
    root = _build_artifact(tmp_path)
    manifest_path = root / "zing_conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_checkpoint"]["local_path"] = "/fsx/private/model.pt"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="private storage marker"):
        validate_artifact(root)
