#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WRAPPER_DIR="$SCRIPT_DIR/real/unitree_sdk2_wrapper"
UV_PROJECT_ARGS=(--project "$PROJECT_ROOT" --python 3.10)

# 1. System deps
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev pybind11-dev

# 2. Python build deps into the repo uv environment
uv pip install "${UV_PROJECT_ARGS[@]}" pybind11 pybind11-stubgen numpy

# 3. Ensure submodule is populated
git -C "$PROJECT_ROOT" submodule update --init deploy/real/unitree_sdk2_wrapper

# 4. Build
cd "$WRAPPER_DIR/python_binding"
export UNITREE_SDK2_PATH="$(pwd)/.."
uv run "${UV_PROJECT_ARGS[@]}" bash build.sh --sdk-path "$UNITREE_SDK2_PATH"

# 5. Install .so into the repo uv environment site-packages
SITE_PACKAGES=$(uv run "${UV_PROJECT_ARGS[@]}" python -c "import site; print(site.getsitepackages()[0])")
echo "Installing to: $SITE_PACKAGES"
cp build/lib/unitree_interface.cpython-*-linux-gnu.so "$SITE_PACKAGES/unitree_interface.so"

# 6. Verify
uv run "${UV_PROJECT_ARGS[@]}" python -c "import unitree_interface; print('unitree_interface installed OK')"
