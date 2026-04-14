"""Convert MJLab tracking-policy NPZ motions into enriched PKL format.

Treats the source NPZ as already enriched motion data and reconstructs the
PKL fields expected by this repo's ``PklMotionLib``.

Usage:
    uv run python -m twist2_mjlab.scripts.npz_to_pkl \
        --input motion.npz \
        --output motion.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
import mujoco as _mujoco

mujoco = cast(Any, _mujoco)

_REQUIRED_KEYS = ("fps", "joint_pos", "body_pos_w", "body_quat_w")
_OPTIONAL_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
_NUM_DOFS = 29


def _load_body_names(xml_path: str) -> list[str]:
  model = mujoco.MjModel.from_xml_path(xml_path)
  body_names = [model.body(i).name for i in range(1, model.nbody)]
  if not body_names:
    raise ValueError("G1 model has no bodies after excluding 'world'.")
  if body_names[0] != "pelvis":
    raise ValueError(
      f"Expected first non-world body to be 'pelvis', got '{body_names[0]}'."
    )
  return body_names


def _require_key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
  if key not in data:
    raise ValueError(f"Missing required NPZ key '{key}'.")
  return np.asarray(data[key])


def _validate_scalar_fps(fps: np.ndarray) -> float:
  fps_flat = np.asarray(fps).reshape(-1)
  if fps_flat.size != 1:
    raise ValueError(f"'fps' must contain exactly one value, got shape {fps.shape}.")
  fps_value = float(fps_flat[0])
  if fps_value <= 0.0:
    raise ValueError(f"'fps' must be positive, got {fps_value}.")
  return fps_value


def _validate_motion_arrays(
  joint_pos: np.ndarray,
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_names: list[str],
) -> int:
  if joint_pos.ndim != 2 or joint_pos.shape[1] != _NUM_DOFS:
    raise ValueError(
      f"'joint_pos' must have shape [T, {_NUM_DOFS}], got {joint_pos.shape}."
    )
  if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
    raise ValueError(f"'body_pos_w' must have shape [T, B, 3], got {body_pos_w.shape}.")
  if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
    raise ValueError(
      f"'body_quat_w' must have shape [T, B, 4], got {body_quat_w.shape}."
    )

  time_steps = joint_pos.shape[0]
  if time_steps == 0:
    raise ValueError("Motion is empty: 'joint_pos' has zero frames.")
  if body_pos_w.shape[0] != time_steps or body_quat_w.shape[0] != time_steps:
    raise ValueError(
      "Time dimension mismatch across required arrays: "
      f"joint_pos={joint_pos.shape}, body_pos_w={body_pos_w.shape}, "
      f"body_quat_w={body_quat_w.shape}."
    )

  expected_bodies = len(body_names)
  if body_pos_w.shape[1] != expected_bodies or body_quat_w.shape[1] != expected_bodies:
    raise ValueError(
      "Body dimension mismatch for G1 model: "
      f"expected {expected_bodies}, got body_pos_w={body_pos_w.shape[1]}, "
      f"body_quat_w={body_quat_w.shape[1]}."
    )

  return time_steps


def _validate_optional_array(
  name: str,
  value: np.ndarray,
  expected_shape: tuple[int, ...],
) -> np.ndarray:
  arr = np.asarray(value)
  if arr.shape != expected_shape:
    raise ValueError(
      f"Optional key '{name}' must have shape {expected_shape}, got {arr.shape}."
    )
  return arr


def convert_npz_to_pkl(
  input_path: str,
  output_path: str,
  xml_path: str,
) -> None:
  body_names = _load_body_names(xml_path)

  with np.load(input_path, allow_pickle=True) as data:
    fps = _validate_scalar_fps(_require_key(data, "fps"))
    joint_pos = _require_key(data, "joint_pos")
    body_pos_w = _require_key(data, "body_pos_w")
    body_quat_w = _require_key(data, "body_quat_w")

    time_steps = _validate_motion_arrays(joint_pos, body_pos_w, body_quat_w, body_names)

    motion: dict[str, Any] = {
      "fps": fps,
      "root_pos": np.asarray(body_pos_w[:, 0, :], dtype=np.float32),
      "root_rot": np.asarray(body_quat_w[:, 0, [1, 2, 3, 0]], dtype=np.float32),
      "dof_pos": np.asarray(joint_pos, dtype=np.float32),
      "link_body_list": body_names,
      "body_pos_w": np.asarray(body_pos_w, dtype=np.float32),
      "body_quat_w": np.asarray(body_quat_w, dtype=np.float32),
      "joint_pos": np.asarray(joint_pos, dtype=np.float32),
    }

    for key in _OPTIONAL_KEYS:
      if key not in data:
        continue
      expected_shape = (time_steps, _NUM_DOFS) if key == "joint_vel" else (time_steps, len(body_names), 3)
      motion[key] = _validate_optional_array(key, data[key], expected_shape).astype(
        np.float32, copy=False
      )

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("wb") as f:
    pickle.dump(motion, f)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert MJLab tracking-policy NPZ motion into enriched PKL."
  )
  parser.add_argument("--input", required=True, help="Path to source motion NPZ")
  parser.add_argument("--output", required=True, help="Path to output PKL")
  parser.add_argument(
    "--xml-path",
    default=str(G1_XML),
    help="Path to Unitree G1 MuJoCo XML (default: bundled G1 XML)",
  )
  args = parser.parse_args()

  convert_npz_to_pkl(args.input, args.output, args.xml_path)
  print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
  main()
