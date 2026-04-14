#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

MOTION_FILE="${HOME}/twist2/seed_g1_enriched_pkl/seed_dataset.yaml"
NUM_ENVS="${TWIST2_NUM_ENVS:-4096}"
VIDEO_INTERVAL="${TWIST2_VIDEO_INTERVAL:-48000}"
VIDEO_LENGTH="${TWIST2_VIDEO_LENGTH:-500}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Twist2-Flat-Unitree-G1 \
  --env.commands.motion.motion-file "${MOTION_FILE}" \
  --env.commands.motion.gpu-cache False \
  --env.commands.motion.cache-capacity-gb 80.0 \
  --agent.experiment-name g1_twist2_seed_flat \
  --agent.run-name g1_twist2_seed_flat \
  --env.scene.num-envs "${NUM_ENVS}" \
  --video True \
  --video-interval "${VIDEO_INTERVAL}" \
  --video-length "${VIDEO_LENGTH}" \
  "$@"
