#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

MOTION_FILE="${TWIST2_MOTION_FILE:-}"
if [[ -z "${MOTION_FILE}" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Enter motion file path (set TWIST2_MOTION_FILE to skip this prompt): " MOTION_FILE
  else
    echo "Error: TWIST2_MOTION_FILE is required when running non-interactively." >&2
    exit 1
  fi
fi

if [[ -z "${MOTION_FILE}" ]]; then
  echo "Error: motion file path is required." >&2
  exit 1
fi

NUM_ENVS="${TWIST2_NUM_ENVS:-4096}"
VIDEO_INTERVAL="${TWIST2_VIDEO_INTERVAL:-48000}"
VIDEO_LENGTH="${TWIST2_VIDEO_LENGTH:-500}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Twist2-Flat-Unitree-G1 \
  --env.commands.motion.motion-file "${MOTION_FILE}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --video True \
  --video-interval "${VIDEO_INTERVAL}" \
  --video-length "${VIDEO_LENGTH}" \
  "$@"
