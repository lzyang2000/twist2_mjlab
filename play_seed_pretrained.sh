#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${TWIST2_SEED_CHECKPOINT:-${SCRIPT_DIR}/resources/pretrained_seed.pt}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: SEED pretrained checkpoint not found at ${CHECKPOINT}" >&2
  exit 1
fi

"${SCRIPT_DIR}/play_twist2.sh" "${CHECKPOINT}" "$@"
