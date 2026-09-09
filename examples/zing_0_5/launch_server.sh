#!/usr/bin/env bash
set -euo pipefail

zing_model_path="${ZING_MODEL_PATH:-./models/Zing-0.5-SGLang}"
zing_profile="${ZING_PROFILE:-highmem}"

server_args=(
  python3 -m sglang.multimodal_gen.runtime.launch_server
  --model-path "${zing_model_path}"
  --pipeline-class-name ZingCausalDMDPipeline
  --attention-backend fa
  --performance-mode speed
  --realtime-vae-backend local
  --host 0.0.0.0
  --port 30000
)

case "${zing_profile}" in
  highmem)
    server_args+=(
      --realtime-causal-kv-cache-num-frames 97
      --realtime-causal-sink-size 9
    )
    ;;
  32g)
    server_args+=(
      --text-encoder-cpu-offload true
      --vae-cpu-offload true
      --realtime-causal-kv-cache-num-frames 33
      --realtime-causal-sink-size 5
    )
    ;;
  *)
    echo "unknown ZING_PROFILE=${zing_profile}; expected highmem or 32g" >&2
    exit 2
    ;;
esac

exec "${server_args[@]}" "$@"
