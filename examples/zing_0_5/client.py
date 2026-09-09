#!/usr/bin/env python3
"""Small Zing-0.5 realtime WebSocket client that saves returned WebP frames."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import msgspec.msgpack
import websockets

ALLOWED_ACTIONS = frozenset("wasdijkl")
WEBP_CONTENT_TYPE = "image/webp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:30000/v1/realtime_video/generate",
    )
    parser.add_argument("--model", default="Zing-0.5")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-update")
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument(
        "--action",
        action="append",
        default=[],
        help="A key held throughout the smoke run; repeat for combined actions",
    )
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--window", type=int)
    parser.add_argument("--sink", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunks < 1:
        raise ValueError("--chunks must be positive")
    actions = {item.lower() for item in args.action}
    unknown = actions - ALLOWED_ACTIONS
    if unknown:
        raise ValueError(f"unknown actions: {sorted(unknown)}")
    args.action = sorted(actions)
    if args.first_frame is not None and not args.first_frame.is_file():
        raise FileNotFoundError(f"first frame does not exist: {args.first_frame}")
    if args.window is not None and args.window != -1 and args.window <= 0:
        raise ValueError("--window must be -1 or positive")
    if args.sink is not None and args.sink < 0:
        raise ValueError("--sink must be non-negative")
    if args.window not in (None, -1) and args.sink is not None:
        if args.sink >= args.window:
            raise ValueError("--sink must be smaller than --window")


def build_init(args: argparse.Namespace) -> dict:
    is_i2v = args.first_frame is not None
    generated_frames = args.chunks * 16 if is_i2v else max(0, args.chunks - 1) * 16
    condition_inputs = {}
    if args.action:
        condition_inputs["camera_actions"] = [args.action] * generated_frames

    request = {
        "type": "init",
        "generation_mode": "i2v" if is_i2v else "t2v",
        "model": args.model,
        "prompt": args.prompt,
        "size": f"{args.width}x{args.height}",
        "fps": args.fps,
        "seed": args.seed,
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": args.chunks,
        "realtime_output_format": "webp",
        "realtime_preview_max_width": args.width,
    }
    if condition_inputs:
        request["condition_inputs"] = condition_inputs
    if is_i2v:
        request["first_frame"] = args.first_frame.read_bytes()
    else:
        request["num_frames"] = 1 + generated_frames
    if args.window is not None:
        request["realtime_causal_kv_cache_num_frames"] = args.window
    if args.sink is not None:
        request["realtime_causal_sink_size"] = args.sink
    return request


def split_encoded_frames(header: dict, payload: bytes) -> list[bytes]:
    if header.get("content_type") != WEBP_CONTENT_TYPE:
        raise ValueError(
            f"expected {WEBP_CONTENT_TYPE}, got {header.get('content_type')!r}"
        )
    lengths = [int(value) for value in header.get("payload_lengths", [])]
    if not lengths or sum(lengths) != len(payload):
        raise ValueError("invalid encoded frame payload lengths")
    frames = []
    offset = 0
    for length in lengths:
        frames.append(payload[offset : offset + length])
        offset += length
    return frames


async def run(args: argparse.Namespace) -> None:
    validate_args(args)
    request = build_init(args)
    args.output.mkdir(parents=True, exist_ok=True)
    public_request = {
        key: value for key, value in request.items() if key != "first_frame"
    }
    (args.output / "request.json").write_text(
        json.dumps(public_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completed_chunks: set[int] = set()
    frame_index = 0
    async with websockets.connect(
        args.ws_url,
        max_size=None,
        open_timeout=args.timeout,
    ) as websocket:
        await websocket.send(msgspec.msgpack.encode(request))
        if args.prompt_update:
            await websocket.send(
                msgspec.msgpack.encode(
                    {
                        "type": "event",
                        "kind": "prompt",
                        "payload": args.prompt_update,
                        "event_id": 1,
                    }
                )
            )

        while len(completed_chunks) < args.chunks:
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
            if message_type == "error":
                raise RuntimeError(header.get("content", "unknown realtime error"))
            if message_type in {"chunk_stats", "chunk_telemetry", "trace"}:
                continue
            if message_type == "frame_batch":
                payload = header.pop("payload")
            elif message_type == "frame_batch_header":
                payload = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            else:
                continue

            for frame in split_encoded_frames(header, payload):
                path = args.output / f"frame-{frame_index:05d}.webp"
                path.write_bytes(frame)
                frame_index += 1
            if header.get("is_final_frame_batch", True):
                completed_chunks.add(int(header["chunk_index"]))

    print(
        json.dumps(
            {
                "chunks": len(completed_chunks),
                "frames": frame_index,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
