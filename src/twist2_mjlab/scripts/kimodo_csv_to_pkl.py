"""Convert Kimodo G1 MuJoCo qpos CSV into enriched TWIST2 PKL format.

Kimodo's G1 export writes MuJoCo qpos rows with shape ``(T, 36)``:
root position ``(3)``, root quaternion ``wxyz`` ``(4)``, and 29 joint angles.

This script runs MuJoCo forward kinematics on each frame and writes the
enriched PKL format expected by ``twist2_mjlab.pkl_motion_lib.PklMotionLib``.

Usage:
    uv run python -m twist2_mjlab.scripts.kimodo_csv_to_pkl \
        --input /home/yiling/kimodo/output.csv \
        --output /home/yiling/kimodo/output.pkl \
        --fps 30
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

_NUM_QPOS_COLS = 36
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


def _load_qpos_csv(input_path: str) -> np.ndarray:
  qpos = np.loadtxt(input_path, delimiter=",", dtype=np.float64)
  if qpos.ndim == 1:
    qpos = qpos[None, :]
  if qpos.ndim != 2 or qpos.shape[1] != _NUM_QPOS_COLS:
    raise ValueError(
      f"Expected Kimodo G1 qpos CSV with shape [T, {_NUM_QPOS_COLS}], got {qpos.shape}."
    )
  if qpos.shape[0] < 2:
    raise ValueError("Motion must contain at least 2 frames.")
  return qpos


def convert_kimodo_csv_to_pkl(
  input_path: str,
  output_path: str,
  fps: float,
  xml_path: str,
) -> None:
  if fps <= 0.0:
    raise ValueError(f"'fps' must be positive, got {fps}.")

  qpos = _load_qpos_csv(input_path)
  body_names = _load_body_names(xml_path)

  root_pos = qpos[:, :3].astype(np.float32, copy=False)
  root_quat_wxyz = qpos[:, 3:7].astype(np.float32, copy=False)
  dof_pos = qpos[:, 7:].astype(np.float32, copy=False)
  if dof_pos.shape[1] != _NUM_DOFS:
    raise ValueError(f"Expected {_NUM_DOFS} DoFs, got {dof_pos.shape[1]}.")

  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)

  n_frames = qpos.shape[0]
  n_bodies = len(body_names)
  body_pos_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((n_frames, n_bodies, 4), dtype=np.float32)
  body_quat_w[:, :, 0] = 1.0

  for t in range(n_frames):
    data.qpos[:3] = root_pos[t]
    data.qpos[3:7] = root_quat_wxyz[t]
    data.qpos[7:] = dof_pos[t]
    mujoco.mj_kinematics(model, data)

    for body_idx in range(n_bodies):
      mj_body_idx = body_idx + 1  # body 0 is world
      body_pos_w[t, body_idx] = data.xpos[mj_body_idx]
      body_quat_w[t, body_idx] = data.xquat[mj_body_idx]

  motion = {
    "fps": float(fps),
    "root_pos": root_pos,
    "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]].copy(),  # xyzw for TWIST2 PKL convention
    "dof_pos": dof_pos,
    "link_body_list": body_names,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "joint_pos": dof_pos.copy(),
  }

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("wb") as f:
    pickle.dump(motion, f)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert Kimodo G1 qpos CSV into enriched TWIST2 PKL."
  )
  parser.add_argument("--input", required=True, help="Path to Kimodo G1 qpos CSV")
  parser.add_argument("--output", required=True, help="Path to output PKL")
  parser.add_argument("--fps", type=float, default=30.0, help="Motion frame rate in Hz")
  parser.add_argument(
    "--xml-path",
    default=str(G1_XML),
    help="Path to Unitree G1 MuJoCo XML (default: bundled G1 XML)",
  )
  args = parser.parse_args()

  convert_kimodo_csv_to_pkl(
    input_path=args.input,
    output_path=args.output,
    fps=args.fps,
    xml_path=args.xml_path,
  )
  print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
  main()
