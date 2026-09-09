# Zing-0.5 on SGLang

This repository is a Seedleap-maintained SGLang inference fork for
[Zing-0.5](https://modelscope.cn/models/seedleap/Zing-0.5). It serves the model
through SGLang's realtime WebSocket API and supports:

- text-to-video and image-initialized video generation;
- W/A/S/D movement and I/J/K/L camera controls;
- prompt changes during an active rollout;
- bounded causal KV caches for long sessions;
- WebP, JPEG, or lossless RGB frame transport.

The internal implementation keeps the historical `MinWM*` Python class names for
checkpoint compatibility. Public model names and documentation use `Zing-0.5`.

## Requirements

- Linux and an NVIDIA CUDA GPU;
- Python 3.11;
- at least 32 GiB GPU memory for the conservative offload profile;
- at least 80 GiB GPU memory for the recommended resident profile.

The memory thresholds are starting profiles, not substitutes for checking the
actual memory reported by the device.

## Install

From this repository:

```bash
python -m pip install -e "python[diffusion]"
```

## Download the serving artifact

```bash
modelscope download \
  --model seedleap/Zing-0.5-SGLang \
  --local_dir ./models/Zing-0.5-SGLang
```

The serving artifact is a sharded safetensors conversion of the public
`seedleap/Zing-0.5` release. Its conversion manifest records the public source
and tensor summary without recording private filesystem paths.

## Launch

The bundled launcher selects the published high-memory profile by default:

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

Recommended profile for an H100/H200-class GPU with at least 80 GiB:

```bash
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path ./models/Zing-0.5-SGLang \
  --pipeline-class-name ZingCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30000
```

For a 32 GiB GPU, begin with CPU offload and the `33/5` cache profile:

```bash
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path ./models/Zing-0.5-SGLang \
  --pipeline-class-name ZingCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --realtime-causal-kv-cache-num-frames 33 \
  --realtime-causal-sink-size 5 \
  --host 0.0.0.0 \
  --port 30000
```

Then send a smoke request:

```bash
python examples/zing_0_5/client.py \
  --prompt "A first-person walk through a misty pine forest at sunrise" \
  --action w \
  --chunks 4 \
  --window 33 \
  --sink 5 \
  --output outputs/forest
```

On an 80 GiB or larger GPU, omit `--window` and `--sink` to use the published
`97/9` profile. Full-history attention remains available with `--window -1
--sink 0`, but its memory use grows with the rollout.

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

## Convert the original public release

If you downloaded the original PyTorch-layout release instead of the serving
artifact:

```bash
python -m sglang.multimodal_gen.tools.convert_zing_checkpoint \
  --zing-dir ./models/Zing-0.5 \
  --output-dir ./models/Zing-0.5-SGLang
```

`--zing-dir` must contain `generator/model.pt` and
`pretrained/{text_encoder,tokenizer,vae}`. The DMD scheduler is created by the
runtime and is not required in either directory. The converter memory-maps the
PyTorch checkpoint and writes one shard at a time, so peak host memory is
bounded by the selected `--max-shard-gib` rather than the full generator size.

Before publishing the converted directory, run the release gate and generate
checksums:

```bash
python -m sglang.multimodal_gen.tools.verify_zing_artifact \
  ./models/Zing-0.5-SGLang \
  --write-sha256
```

The verifier rejects missing or mismatched shards, symlinks, unexpected model
identifiers, and private storage paths in the conversion metadata.

## License and provenance

The fork is Apache-2.0. See [THIRD_PARTY_NOTICES_ZING.md](THIRD_PARTY_NOTICES_ZING.md)
for Zing, minWM, Wan, and optional dependency provenance. Model weights remain
subject to the license and notices published with their model repository.
