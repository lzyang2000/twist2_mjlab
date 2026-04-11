#!/usr/bin/env bash
# TWIST2 sim2sim launcher.
# No ROS dependency — just MuJoCo physics + ONNX policy + motion library.
#
# Usage:
#   TWIST2_MOTION_FILE=/path/to/motion.pkl ./deploy/play_sim_twist2.sh [/path/to/model.onnx|model.pt]
#
# Accepts .onnx directly or .pt checkpoint (auto-exports to ONNX).
# If no path is given, finds the latest checkpoint and exports automatically.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
CLEANED_UP=0

cleanup() {
  if [[ "${CLEANED_UP}" == "1" ]]; then
    return
  fi
  CLEANED_UP=1
  trap '' SIGINT SIGTERM EXIT
  echo -e '\nStopping...'
  pkill -f "deploy/policy/twist2_policy.py" 2>/dev/null || true
  pkill -f "deploy/sim/sim_node.py"         2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 1. Resolve motion file
# ---------------------------------------------------------------------------
MOTION_FILE="${TWIST2_MOTION_FILE:-}"
if [[ -z "${MOTION_FILE}" ]]; then
  echo "Error: TWIST2_MOTION_FILE not set."
  echo "  export TWIST2_MOTION_FILE=/path/to/motion.pkl"
  exit 1
fi
if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "Error: Motion file not found: ${MOTION_FILE}"
  exit 1
fi
echo "Motion file: ${MOTION_FILE}"

# ---------------------------------------------------------------------------
# 2. Resolve ONNX model
# ---------------------------------------------------------------------------
MODEL_ARG="${1:-}"
MOTION_INDEX="${TWIST2_MOTION_INDEX:-0}"

_export_onnx_from_pt() {
  local ckpt="$1"
  echo "Exporting ONNX from: ${ckpt}"
  TWIST2_MOTION_FILE="${MOTION_FILE}" uv run python deploy/export_onnx.py "${ckpt}"
  local run_dir
  run_dir="$(dirname "${ckpt}")"
  ONNX_MODEL="$(ls -t "${run_dir}"/*.onnx 2>/dev/null | head -1)"
  if [[ -z "${ONNX_MODEL}" ]]; then
    echo "Error: ONNX export failed." >&2
    exit 1
  fi
}

if [[ -z "${MODEL_ARG}" ]]; then
  # No arg: find latest checkpoint automatically
  EXPERIMENT_DIR="${ROOT_DIR}/logs/rsl_rl/g1_twist2_flat"
  LATEST_RUN="$(ls -dt "${EXPERIMENT_DIR}"/*/ 2>/dev/null | head -1)"
  if [[ -z "${LATEST_RUN}" ]]; then
    echo "Error: No runs found in ${EXPERIMENT_DIR}/" >&2
    exit 1
  fi
  LATEST_CKPT="$(ls "${LATEST_RUN}"model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "${LATEST_CKPT}" ]]; then
    echo "Error: No .pt checkpoints found in ${LATEST_RUN}" >&2
    exit 1
  fi
  _export_onnx_from_pt "${LATEST_CKPT}"
elif [[ "${MODEL_ARG}" == *.pt ]]; then
  # .pt checkpoint: export to ONNX first
  if [[ ! -f "${MODEL_ARG}" ]]; then
    echo "Error: Checkpoint not found: ${MODEL_ARG}" >&2
    exit 1
  fi
  _export_onnx_from_pt "${MODEL_ARG}"
  shift
elif [[ "${MODEL_ARG}" == *.onnx ]]; then
  # .onnx: use directly
  if [[ ! -f "${MODEL_ARG}" ]]; then
    echo "Error: ONNX file not found: ${MODEL_ARG}" >&2
    exit 1
  fi
  ONNX_MODEL="${MODEL_ARG}"
  shift
else
  echo "Error: Unsupported file type: ${MODEL_ARG} (expected .pt or .onnx)" >&2
  exit 1
fi
echo "ONNX model: ${ONNX_MODEL}"

trap cleanup SIGINT SIGTERM EXIT

cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 3. PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}:${ROOT_DIR}/src"

# ---------------------------------------------------------------------------
# 4. Launch nodes
# ---------------------------------------------------------------------------

# Policy (background)
echo "Starting TWIST2 Policy Node (in background)..."
uv run python deploy/policy/twist2_policy.py "${ONNX_MODEL}" \
  --motion-file "${MOTION_FILE}" \
  --motion-index "${MOTION_INDEX}" &

sleep 2.0

# Sim (foreground)
echo "Starting Simulation Node (in foreground)..."
uv run python deploy/sim/sim_node.py
