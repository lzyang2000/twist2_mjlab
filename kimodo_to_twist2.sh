#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMODO_DIR="${KIMODO_DIR:-$(cd "${SCRIPT_DIR}/../kimodo" && pwd)}"

MODEL="${KIMODO_MODEL:-Kimodo-G1-RP-v1}"
DURATION="${KIMODO_DURATION:-5.0}"
FPS="${TWIST2_MOTION_FPS:-30}"
OUTPUT_DIR="${KIMODO_TWIST2_OUTPUT_DIR:-${SCRIPT_DIR}/generated/kimodo_bridge}"
CHECKPOINT="${TWIST2_CHECKPOINT:-}"
PLAY_MODE="pretrained"
FORCE=0

run_kimodo_python_module() {
  python -m "$@"
}

usage() {
  cat <<EOF
Usage:
  $0 "text prompt" [options] [-- play_args...]

Options:
  --model NAME         Kimodo model name (default: ${MODEL})
  --duration SEC       Motion duration in seconds (default: ${DURATION})
  --fps FPS            Motion FPS for TWIST2 PKL (default: ${FPS})
  --output-dir DIR     Output directory for generated files
  --checkpoint PATH    Use this TWIST2 checkpoint instead of pretrained
  --latest             Use latest TWIST2 checkpoint instead of pretrained
  --no-play            Stop after generating CSV/PKL
  --force              Overwrite existing outputs for the same prompt hash
  -h, --help           Show this help

Environment overrides:
  KIMODO_DIR
  KIMODO_MODEL
  KIMODO_DURATION
  KIMODO_TWIST2_OUTPUT_DIR
  TWIST2_MOTION_FPS
  TWIST2_CHECKPOINT

Notes:
  - This is a single-command prompt -> Kimodo generation -> TWIST2 playback bridge.
  - It is not streaming generation; Kimodo still generates the full clip first.
  - If kimodo_textencoder is already running, generation will be faster.
  - Activate the Kimodo env before running this script.
  - Kimodo is invoked via: python -m kimodo.scripts.generate
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

PROMPT=""
PLAY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      PLAY_MODE="checkpoint"
      shift 2
      ;;
    --latest)
      PLAY_MODE="latest"
      shift
      ;;
    --no-play)
      PLAY_MODE="none"
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --)
      shift
      PLAY_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "${PROMPT}" ]]; then
        PROMPT="$1"
        shift
      else
        echo "Unexpected positional argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${PROMPT}" ]]; then
  echo "Error: text prompt is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${KIMODO_DIR}" ]]; then
  echo "Error: Kimodo repo not found at ${KIMODO_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PROMPT_HASH="$(printf '%s' "${PROMPT}" | sha1sum | cut -c1-12)"
STEM="${OUTPUT_DIR}/motion_${PROMPT_HASH}"
CSV_PATH="${STEM}.csv"
NPZ_PATH="${STEM}.npz"
PKL_PATH="${STEM}.pkl"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"

if [[ ${FORCE} -eq 1 || ! -f "${CSV_PATH}" || ! -f "${NPZ_PATH}" ]]; then
  echo "[1/3] Generating G1 motion with Kimodo"
  (
    cd "${KIMODO_DIR}"
    run_kimodo_python_module kimodo.scripts.generate "${PROMPT}" \
      --model "${MODEL}" \
      --duration "${DURATION}" \
      --output "${STEM}"
  )
else
  echo "[1/3] Reusing existing Kimodo outputs:"
  echo "      ${CSV_PATH}"
  echo "      ${NPZ_PATH}"
fi

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "Error: expected Kimodo CSV output at ${CSV_PATH}" >&2
  exit 1
fi

if [[ ${FORCE} -eq 1 || ! -f "${PKL_PATH}" ]]; then
  echo "[2/3] Converting Kimodo CSV to TWIST2 PKL"
  (
    cd "${SCRIPT_DIR}"
    uv run python -m twist2_mjlab.scripts.kimodo_csv_to_pkl \
      --input "${CSV_PATH}" \
      --output "${PKL_PATH}" \
      --fps "${FPS}"
  )
else
  echo "[2/3] Reusing existing TWIST2 PKL:"
  echo "      ${PKL_PATH}"
fi

echo "Generated files:"
echo "  NPZ: ${NPZ_PATH}"
echo "  CSV: ${CSV_PATH}"
echo "  PKL: ${PKL_PATH}"

if [[ "${PLAY_MODE}" == "none" ]]; then
  echo "[3/3] Skipping TWIST2 playback (--no-play)"
  exit 0
fi

echo "[3/3] Launching TWIST2 playback"
if [[ "${PLAY_MODE}" == "checkpoint" ]]; then
  TWIST2_MOTION_FILE="${PKL_PATH}" "${SCRIPT_DIR}/play_twist2.sh" "${CHECKPOINT}" "${PLAY_ARGS[@]}"
elif [[ "${PLAY_MODE}" == "latest" ]]; then
  TWIST2_MOTION_FILE="${PKL_PATH}" "${SCRIPT_DIR}/play_twist2.sh" "${PLAY_ARGS[@]}"
else
  TWIST2_MOTION_FILE="${PKL_PATH}" "${SCRIPT_DIR}/play_twist2_pretrained.sh" "${PLAY_ARGS[@]}"
fi
