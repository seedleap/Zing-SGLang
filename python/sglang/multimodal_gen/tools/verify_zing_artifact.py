# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0
"""Validate a converted Zing-0.5 serving artifact before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open

EXPECTED_COMPONENTS = ("text_encoder", "tokenizer", "vae")
EXPECTED_PIPELINE = "ZingCausalDMDPipeline"
EXPECTED_SOURCE_URIS = {
    "modelscope://seedleap/Zing-0.5",
    "https://modelscope.cn/models/seedleap/Zing-0.5",
}
PRIVATE_MARKERS = (
    "s3://",
    "/fsx/",
    "amazonaws.com",
    ".internal",
    "ecr.",
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _validate_public_metadata(metadata: dict[str, Any]) -> None:
    for value in _walk_strings(metadata):
        lowered = value.lower()
        if any(marker in lowered for marker in PRIVATE_MARKERS):
            raise ValueError(f"private storage marker found in metadata: {value!r}")
        if value.startswith("/") or WINDOWS_ABSOLUTE_PATH.match(value):
            raise ValueError(f"local absolute path found in metadata: {value!r}")


def _validate_no_symlinks(root: Path) -> None:
    symlinks = [path.relative_to(root) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        rendered = ", ".join(str(path) for path in symlinks[:8])
        raise ValueError(f"artifact must not contain symlinks: {rendered}")


def _validate_component(root: Path, component: str) -> int:
    path = root / component
    if not path.is_dir():
        raise ValueError(f"missing required component directory: {path}")
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    if not files:
        raise ValueError(f"component directory is empty: {path}")
    return len(files)


def _validate_shards(transformer_dir: Path, index: dict[str, Any]) -> dict[str, int]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("transformer index has no weight_map")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("transformer weight_map entries must be strings")

    referenced_names = set(weight_map.values())
    shard_names = {
        path.name
        for path in transformer_dir.glob("diffusion_pytorch_model-*.safetensors")
    }
    if shard_names != referenced_names:
        missing = sorted(referenced_names - shard_names)
        extra = sorted(shard_names - referenced_names)
        raise ValueError(
            f"safetensors shard mismatch; missing={missing}, extra={extra}"
        )

    seen_keys: set[str] = set()
    for shard_name in sorted(shard_names):
        shard_path = transformer_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
        expected_keys = {
            key for key, value in weight_map.items() if value == shard_name
        }
        if shard_keys != expected_keys:
            missing = sorted(expected_keys - shard_keys)
            extra = sorted(shard_keys - expected_keys)
            raise ValueError(
                f"weight index mismatch for {shard_name}; missing={missing[:8]}, "
                f"extra={extra[:8]}"
            )
        overlap = seen_keys.intersection(shard_keys)
        if overlap:
            raise ValueError(
                f"duplicate tensor keys across shards: {sorted(overlap)[:8]}"
            )
        seen_keys.update(shard_keys)

    total_size = index.get("metadata", {}).get("total_size")
    if not isinstance(total_size, int) or total_size <= 0:
        raise ValueError("transformer index metadata.total_size must be positive")
    return {
        "shard_count": len(shard_names),
        "tensor_count": len(seen_keys),
        "tensor_bytes": total_size,
    }


def validate_artifact(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    _validate_no_symlinks(root)

    model_index = _load_json(root / "model_index.json")
    if model_index.get("_class_name") != EXPECTED_PIPELINE:
        raise ValueError(f"model_index.json must use _class_name={EXPECTED_PIPELINE!r}")
    if model_index.get("scheduler", object()) is not None:
        raise ValueError("model_index.json scheduler must be null")
    for component in EXPECTED_COMPONENTS:
        if component not in model_index:
            raise ValueError(f"model_index.json is missing {component!r}")

    transformer_dir = root / "transformer"
    transformer_config = _load_json(transformer_dir / "config.json")
    if transformer_config.get("_class_name") != "MinWMCausalTransformer3DModel":
        raise ValueError("unexpected transformer class in transformer/config.json")
    local_attn_size = transformer_config.get("local_attn_size")
    sink_size = transformer_config.get("sink_size")
    sliding_window = transformer_config.get("sliding_window_num_frames")
    if local_attn_size != -1 and sliding_window != local_attn_size:
        raise ValueError("bounded cache window does not match local_attn_size")
    if not isinstance(sink_size, int) or sink_size < 0:
        raise ValueError("transformer sink_size must be a non-negative integer")
    if local_attn_size != -1 and sink_size >= local_attn_size:
        raise ValueError("transformer sink_size must be smaller than local_attn_size")

    shard_index = _load_json(
        transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    )
    shard_summary = _validate_shards(transformer_dir, shard_index)

    manifest = _load_json(root / "zing_conversion_manifest.json")
    _validate_public_metadata(manifest)
    if manifest.get("format") != "sglang-zing-0.5-v1":
        raise ValueError("unexpected conversion manifest format")
    source_uri = manifest.get("source_checkpoint", {}).get("uri")
    if source_uri not in EXPECTED_SOURCE_URIS:
        raise ValueError(f"unexpected public source URI: {source_uri!r}")
    if manifest.get("components") != list(EXPECTED_COMPONENTS):
        raise ValueError("conversion manifest component list is not canonical")
    if (
        manifest.get("safetensors", {}).get("total_size")
        != shard_summary["tensor_bytes"]
    ):
        raise ValueError("manifest and transformer index disagree on tensor byte size")
    if (
        manifest.get("generator", {}).get("tensor_count")
        != shard_summary["tensor_count"]
    ):
        raise ValueError("manifest and transformer index disagree on tensor count")

    component_files = {
        component: _validate_component(root, component)
        for component in EXPECTED_COMPONENTS
    }
    return {
        "artifact": str(root),
        "pipeline": EXPECTED_PIPELINE,
        "source": source_uri,
        "cache": {"window": local_attn_size, "sink": sink_size},
        "component_files": component_files,
        **shard_summary,
    }


def write_sha256sums(root: Path) -> int:
    output = root / "SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != output and not path.is_symlink()
    )
    lines = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--write-sha256",
        action="store_true",
        help="Write SHA256SUMS for every regular artifact file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_artifact(args.artifact)
    if args.write_sha256:
        summary["hashed_files"] = write_sha256sums(args.artifact.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
