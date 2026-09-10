#!/usr/bin/env bash
set -euo pipefail

zing_model_path="${ZING_MODEL_PATH:-./models/Zing-0.5-SGLang}"
zing_taehv_path="${ZING_TAEHV_PATH:-${zing_model_path}/taew2_2.pth}"
zing_profile="${ZING_PROFILE:-highmem}"
zing_vae_lane=""

server_args=(
  python3 -m sglang.multimodal_gen.runtime.launch_server
  --model-path "${zing_model_path}"
  --pipeline-class-name ZingCausalDMDPipeline
  --attention-backend fa
  --performance-mode speed
  --host 0.0.0.0
  --port 30000
)

case "${zing_profile}" in
  highmem)
    zing_vae_lane="parity"
    server_args+=(
      --realtime-causal-kv-cache-num-frames 97
      --realtime-causal-sink-size 9
    )
    ;;
  32g)
    if [[ ! -f "${zing_taehv_path}" ]]; then
      echo "TAEHV checkpoint not found: ${zing_taehv_path}" >&2
      echo "Set ZING_TAEHV_PATH or place taew2_2.pth in the model directory." >&2
      exit 2
    fi
    zing_vae_lane="parallel"
    server_args+=(
      --text-encoder-cpu-offload true
      --vae-cpu-offload true
      --vae-config.taehv-checkpoint-path "${zing_taehv_path}"
      --realtime-causal-kv-cache-num-frames 32
      --realtime-causal-sink-size 8
    )
    ;;
  *)
    echo "unknown ZING_PROFILE=${zing_profile}; expected highmem or 32g" >&2
    exit 2
    ;;
esac

MINWM_VAE_LANE="${zing_vae_lane}" exec "${server_args[@]}" "$@"
