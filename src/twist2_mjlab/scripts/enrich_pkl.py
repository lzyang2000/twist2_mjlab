"""Enrich PKL motion files with world-frame body positions and orientations.

Takes a TWIST2 YAML dataset file, runs MuJoCo forward kinematics on every
referenced PKL to compute body_pos_w and body_quat_w, and outputs enriched
PKLs + a new YAML pointing to them.

Usage:
    uv run python -m twist2_mjlab.scripts.enrich_pkl \
        --dataset /path/to/twist2_dataset.yaml \
        --output-dir /path/to/enriched/
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import types
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
import mujoco as _mujoco

mujoco = cast(Any, _mujoco)

# Ensure numpy._core.multiarray exists for unpickling old PKL files.
# Modern numpy already has it; only patch if truly missing.
if "numpy._core" not in sys.modules:
  sys.modules["numpy._core"] = types.ModuleType("numpy._core")
if "numpy._core.multiarray" not in sys.modules:
  _mc = types.ModuleType("numpy._core.multiarray")
  _mc._reconstruct = np.core.multiarray._reconstruct  # type: ignore[attr-defined]
  sys.modules["numpy._core.multiarray"] = _mc


def _build_body_name_map(
  model: Any, pkl_link_body_list: list[str]
) -> dict[int, int]:
  """Map PKL link_body_list indices → MuJoCo body indices, by name."""
  mj_name_to_idx: dict[str, int] = {}
  for i in range(model.nbody):
    mj_name_to_idx[model.body(i).name] = i

  pkl_to_mj: dict[int, int] = {}
  for pkl_idx, name in enumerate(pkl_link_body_list):
    if name in mj_name_to_idx:
      pkl_to_mj[pkl_idx] = mj_name_to_idx[name]
  return pkl_to_mj


def enrich_single_pkl(
  input_path: str,
  output_path: str,
  xml_path: str,
) -> str | None:
  """Enrich a single PKL file with MuJoCo FK data. Returns error string or None."""
  try:
    with open(input_path, "rb") as f:
      data = pickle.load(f)
  except Exception as e:
    return f"Failed to load {input_path}: {e}"

  root_pos = np.asarray(data["root_pos"])  # [T, 3]
  root_rot = np.asarray(data["root_rot"])  # [T, 4] in [x,y,z,w]
  dof_pos = np.asarray(data["dof_pos"])  # [T, 29]
  link_body_list = data["link_body_list"]
  T = root_pos.shape[0]

  # Convert root_rot from [x,y,z,w] → [w,x,y,z] for MuJoCo
  root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]

  model = mujoco.MjModel.from_xml_path(xml_path)
  mj_data = mujoco.MjData(model)
  pkl_to_mj = _build_body_name_map(model, link_body_list)

  n_bodies = len(link_body_list)
  body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)
  # Default quaternion: identity [w,x,y,z] = [1,0,0,0]
  body_quat_w[:, :, 0] = 1.0

  for t in range(T):
    mj_data.qpos[:3] = root_pos[t]
    mj_data.qpos[3:7] = root_rot_wxyz[t]
    mj_data.qpos[7:] = dof_pos[t]
    mujoco.mj_kinematics(model, mj_data)

    for pkl_idx, mj_idx in pkl_to_mj.items():
      body_pos_w[t, pkl_idx] = mj_data.xpos[mj_idx]
      body_quat_w[t, pkl_idx] = mj_data.xquat[mj_idx]

  # Save enriched PKL (preserve all original keys + add new ones)
  enriched = dict(data)
  enriched["body_pos_w"] = body_pos_w
  enriched["body_quat_w"] = body_quat_w

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, "wb") as f:
    pickle.dump(enriched, f)

  return None


def _worker(args: tuple[str, str, str]) -> tuple[str, str | None]:
  """Multiprocessing worker. Returns (input_path, error_or_none)."""
  input_path, output_path, xml_path = args
  err = enrich_single_pkl(input_path, output_path, xml_path)
  return (input_path, err)


def main() -> None:
  parser = argparse.ArgumentParser(description="Enrich PKL dataset with MuJoCo FK")
  parser.add_argument(
    "--dataset", required=True, help="Path to TWIST2 dataset YAML file"
  )
  parser.add_argument(
    "--output-dir", required=True, help="Output directory for enriched PKLs + YAML"
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=8,
    help="Number of parallel workers (default: 8)",
  )
  args = parser.parse_args()

  with open(args.dataset) as f:
    config = yaml.safe_load(f)

  root_path = config["root_path"]
  motions = config["motions"]
  output_dir = Path(args.output_dir)
  xml_path = str(G1_XML)

  # Build work list
  work_items: list[tuple[str, str, str]] = []
  for entry in motions:
    rel_file = entry["file"]
    input_path = os.path.join(root_path, rel_file)
    output_path = str(output_dir / rel_file)
    work_items.append((input_path, output_path, xml_path))

  print(f"Enriching {len(work_items)} PKL files → {output_dir}")

  # Process in parallel
  errors: list[str] = []
  n_workers = min(args.workers, len(work_items))
  with Pool(processes=n_workers) as pool:
    for input_path, err in tqdm(
      pool.imap_unordered(_worker, work_items, chunksize=16),
      total=len(work_items),
      desc="Enriching PKLs",
    ):
      if err is not None:
        errors.append(err)

  if errors:
    print(f"\n{len(errors)} errors:")
    for e in errors[:20]:
      print(f"  {e}")
    if len(errors) > 20:
      print(f"  ... and {len(errors) - 20} more")

  # Write new YAML pointing to enriched PKLs
  new_config = {
    "root_path": str(output_dir),
    "motions": motions,  # same entries, new root_path
  }
  output_yaml = output_dir / "dataset.yaml"
  output_dir.mkdir(parents=True, exist_ok=True)
  with open(output_yaml, "w") as f:
    yaml.dump(new_config, f, default_flow_style=False)

  n_ok = len(work_items) - len(errors)
  print(f"\nDone: {n_ok}/{len(work_items)} enriched. YAML: {output_yaml}")


if __name__ == "__main__":
  main()
