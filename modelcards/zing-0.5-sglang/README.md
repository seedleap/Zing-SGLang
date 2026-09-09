---
license: apache-2.0
frameworks:
  - PyTorch
tasks:
  - text-to-video-synthesis
base_model:
  - seedleap/Zing-0.5
---

# Zing-0.5-SGLang

This repository contains the SGLang serving layout of
[Seedleap Zing-0.5](https://modelscope.cn/models/seedleap/Zing-0.5).
The model weights are unchanged numerically and are stored as sharded
safetensors for memory-efficient loading.

## Source

- Original model: `seedleap/Zing-0.5`
- Serving runtime: the Seedleap Zing SGLang inference fork
- License: Apache-2.0, subject to the notices accompanying the original model

`zing_conversion_manifest.json` records the source identifier, tensor count,
parameter count, and total safetensors size. It intentionally contains no local
filesystem paths or private storage locations. `SHA256SUMS` covers every file
in the published artifact.

## Layout

```text
Zing-0.5-SGLang/
├── model_index.json
├── zing_conversion_manifest.json
├── sglang-runtime/
│   ├── ZING.md
│   ├── examples/zing_0_5/
│   └── python/sglang/
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── diffusion_pytorch_model-*.safetensors
├── text_encoder/
├── tokenizer/
└── vae/
```

## Usage

The matching SGLang serving source is bundled in `sglang-runtime/`. Install it
and start the server from the downloaded model directory:

```bash
cd sglang-runtime
python -m pip install -e "python[diffusion]"

ZING_MODEL_PATH=.. examples/zing_0_5/launch_server.sh
```

See `sglang-runtime/ZING.md` for the full launch and client guide. The
high-memory profile uses a `97/9` causal attention window/sink. For a 32 GiB
GPU, begin with `33/5` and CPU offload for the text encoder and VAE.

The runtime creates the four-step DMD scheduler directly, so this artifact does
not contain a `scheduler/` directory.

The public pipeline class is `ZingCausalDMDPipeline`. Historical `MinWM*` class
names remain aliases in the source repository for compatibility.
