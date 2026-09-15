# Zing-0.5 on SGLang

This repository is a Seedleap-maintained SGLang inference fork for
[Zing-0.5](https://modelscope.cn/models/seedleap/Zing-0.5). The reference
PyTorch inference implementation is available in
[zing-world-model](https://github.com/seedleap/zing-world-model). This fork
serves the model through SGLang's realtime WebSocket API and supports:

- text-to-video and image-initialized video generation;
- W/A/S/D movement and I/J/K/L camera controls;
- prompt changes during an active rollout;
- bounded causal KV caches for long sessions;
- WebP, JPEG, or lossless RGB frame transport.

## Requirements

- Linux and an NVIDIA CUDA GPU;
- Python 3.11;
- an H100/H200-class GPU for the resident profile, or begin with the `32g`
  offload profile on a 32 GiB GPU.

The memory thresholds are starting profiles, not substitutes for checking the
actual memory reported by the device.

## Install

From this repository:

```bash
SGLANG_BUILD_RUST_EXTS=none \
  python -m pip install -e "python[diffusion]"
```

Both launch profiles use the optional TAEHV decoder:

```bash
python -m pip install \
  "taehv @ git+https://github.com/madebyollin/taehv.git@093b918971d59001a0bad6dfd6e0409b5e1752cf"
```

## Download the serving artifact

```bash
modelscope download \
  --model seedleap/Zing-0.5-SGLang \
  --local_dir ./models/Zing-0.5-SGLang
```

Download its pinned checkpoint separately:

```bash
curl -L \
  https://raw.githubusercontent.com/madebyollin/taehv/093b918971d59001a0bad6dfd6e0409b5e1752cf/taew2_2.pth \
  -o ./models/Zing-0.5-SGLang/taew2_2.pth

echo "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a  ./models/Zing-0.5-SGLang/taew2_2.pth" \
  | sha256sum --check
```

The public `seedleap/Zing-0.5` release stores its generator as
`generator/model.pt`; its text encoder and VAE already use safetensors. This
serving repository converts only the generator to sharded safetensors for
SGLang loading. Tensor values are unchanged; the file layout and loading
metadata differ. `zing_conversion_manifest.json` records the public source and
tensor summary without private filesystem paths.

Model and pipeline identifiers consistently use `Zing` throughout the
implementation.

## Launch

The bundled launcher uses the local TAEHV decoder and the online cache/RoPE
configuration: `block_relative`, maximum frame gap `12`, window `32`, sink `8`,
and first-frame prompt pinning.

```bash
ZING_MODEL_PATH=./models/Zing-0.5-SGLang \
  examples/zing_0_5/launch_server.sh
```

For the conservative 32 GiB profile:

```bash
ZING_MODEL_PATH=./models/Zing-0.5-SGLang \
ZING_PROFILE=32g \
  examples/zing_0_5/launch_server.sh
```

The equivalent explicit commands are shown below.

Resident profile for an H100/H200-class GPU:

```bash
ZING_VAE_LANE=parallel \
  python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path ./models/Zing-0.5-SGLang \
  --pipeline-class-name ZingCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --vae-config.taehv-checkpoint-path ./models/Zing-0.5-SGLang/taew2_2.pth \
  --realtime-causal-kv-cache-num-frames 32 \
  --realtime-causal-sink-size 8 \
  --host 0.0.0.0 \
  --port 30000
```

For a 32 GiB GPU, begin with CPU offload:

```bash
ZING_VAE_LANE=parallel \
  python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path ./models/Zing-0.5-SGLang \
  --pipeline-class-name ZingCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --vae-config.taehv-checkpoint-path ./models/Zing-0.5-SGLang/taew2_2.pth \
  --realtime-causal-kv-cache-num-frames 32 \
  --realtime-causal-sink-size 8 \
  --host 0.0.0.0 \
  --port 30000
```

Then send a smoke request:

```bash
python examples/zing_0_5/client.py \
  --prompt "A first-person walk through a misty pine forest at sunrise" \
  --action w \
  --chunks 4 \
  --output outputs/forest
```

The client inherits the server's `32/8` cache defaults unless `--window` or
`--sink` is explicitly supplied.

For I2V, add a local PNG, JPEG, or WebP image:

```bash
python examples/zing_0_5/client.py \
  --first-frame ./first-frame.png \
  --prompt "Move forward slowly while preserving the scene" \
  --action w \
  --output outputs/i2v
```

The example writes one WebP image per returned frame plus `request.json`. The
static browser demo under
`python/sglang/multimodal_gen/apps/realtime_webui/` can be served with any local
HTTP server.

## License and provenance

The fork is Apache-2.0. See [THIRD_PARTY_NOTICES_ZING.md](THIRD_PARTY_NOTICES_ZING.md)
for Zing, Wan, and optional dependency provenance. Model weights remain
subject to the license and notices published with their model repository.
