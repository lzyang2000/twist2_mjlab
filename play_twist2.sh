#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="logs/rsl_rl/g1_twist2_flat"
MOTION_FILE="${TWIST2_MOTION_FILE:-}"
CHECKPOINT="${1:-}"

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

if [[ -z "${CHECKPOINT}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  LATEST_RUN="$(ls -dt "${SCRIPT_DIR}/${EXPERIMENT_DIR}"/*/ 2>/dev/null | head -1)"
  if [[ -z "${LATEST_RUN}" ]]; then
    echo "Error: No runs found in ${EXPERIMENT_DIR}/" >&2
    exit 1
  fi
  CHECKPOINT="$(ls -t "${LATEST_RUN}"model_*.pt 2>/dev/null | head -1)"
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "Error: No checkpoints found in ${LATEST_RUN}" >&2
    exit 1
  fi
  echo "Auto-selected: ${CHECKPOINT}"
else
  shift
fi

uv run play Twist2-Flat-Unitree-G1 \
  --checkpoint-file "${CHECKPOINT}" \
  --motion-file "${MOTION_FILE}" \
  --num-envs 1 \
  --device cpu \
  --viewer native \
  "$@"
