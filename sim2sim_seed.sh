#!/usr/bin/env bash
# TWIST2 Seed sim2sim launcher.
#
# Usage:
#   TWIST2_MOTION_FILE=/path/to/single_motion.pkl ./sim2sim_seed.sh [model.onnx|model.pt]
#
# TWIST2_MOTION_FILE is needed for sim2sim (single PKL for the motion index).
# If no model arg is given, finds the latest checkpoint and exports ONNX.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/logs/rsl_rl/g1_twist2_seed_flat"

MOTION_FILE="${TWIST2_MOTION_FILE:-}"
if [[ -z "${MOTION_FILE}" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Enter motion PKL path for sim2sim: " MOTION_FILE
  else
    echo "Error: TWIST2_MOTION_FILE is required for sim2sim." >&2
    exit 1
  fi
fi

if [[ -z "${MOTION_FILE}" ]]; then
  echo "Error: motion file path is required." >&2
  exit 1
fi

export TWIST2_MOTION_FILE="${MOTION_FILE}"

MODEL_ARG="${1:-}"

if [[ -z "${MODEL_ARG}" ]]; then
  LATEST_RUN="$(ls -dt "${EXPERIMENT_DIR}"/*/ 2>/dev/null | head -1)"
  if [[ -z "${LATEST_RUN}" ]]; then
    echo "Error: No runs found in ${EXPERIMENT_DIR}/" >&2
    exit 1
  fi
  MODEL_ARG="$(ls "${LATEST_RUN}"model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "${MODEL_ARG}" ]]; then
    echo "Error: No checkpoints found in ${LATEST_RUN}" >&2
    exit 1
  fi
  echo "Auto-selected: ${MODEL_ARG}"
else
  shift
fi

exec "${SCRIPT_DIR}/deploy/play_sim_twist2.sh" "${MODEL_ARG}" "$@"
