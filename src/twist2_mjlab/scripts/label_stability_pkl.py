"""Add per-frame stability labels to enriched PKL motion files.

Reads a dataset YAML produced by enrich_pkl.py (must contain body_pos_w),
computes three per-frame stability metrics for each PKL, writes labelled PKLs
to an output directory, and produces a new YAML with per-motion summary stats.

Labels added to each PKL
------------------------
foot_contact          [T, 2] bool   left / right foot on ground
com_pos_w             [T, 3] f32    whole-body centre of mass (world frame)
com_vel_w             [T, 3] f32    CoM velocity (central finite-difference)
capture_point_xy      [T, 2] f32    LIPM capture point  CP = CoM_xy + v_xy/ω
is_statically_stable  [T]    bool   CoM inside active-contact support polygon
is_step_viable        [T]    bool   CP  inside active + projected-swing polygon
stability_margin      [T]    f32    min signed dist to either polygon
                                    negative = inside (stable), positive = outside

Per-motion stats added to the output YAML
-----------------------------------------
static_stable_ratio   fraction of frames where is_statically_stable is True
step_viable_ratio     fraction of frames where is_step_viable is True
falling_ratio         fraction of frames labelled as genuinely falling
mean_stability_margin mean of stability_margin across the clip

Usage
-----
    uv run python -m twist2_mjlab.scripts.label_stability_pkl \\
        --dataset /path/to/enriched/dataset.yaml \\
        --output-dir /path/to/labelled/ \\
        --workers 8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import types
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

import mujoco as _mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

# --------------------------------------------------------------------------
# numpy compatibility shim (same as enrich_pkl.py)
# --------------------------------------------------------------------------
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = types.ModuleType("numpy._core")
if "numpy._core.multiarray" not in sys.modules:
    _mc = types.ModuleType("numpy._core.multiarray")
    _mc._reconstruct = np.core.multiarray._reconstruct  # type: ignore[attr-defined]
    sys.modules["numpy._core.multiarray"] = _mc

# --------------------------------------------------------------------------
# G1 body masses (loaded once at module level, shared across workers)
# --------------------------------------------------------------------------
_G1_MODEL = _mujoco.MjModel.from_xml_path(str(G1_XML))
_G1_BODY_NAMES: list[str] = [
    _mujoco.mj_id2name(_G1_MODEL, _mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(_G1_MODEL.nbody)
]
_G1_MASSES: np.ndarray = _G1_MODEL.body_mass.copy()  # [nbody]
_G1_TOTAL_MASS: float = float(_G1_MASSES.sum())

# Physics constants
_G: float = 9.81          # m/s²
_FOOT_CONTACT_THRESH: float = 0.08   # m — ankle roll link z below this → contact
_FOOT_PAD: float = 0.12              # m — half-foot padding around contact point
_FALLING_MARGIN: float = 0.15        # m — CP this far outside polygon → falling


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _dist_point_to_segment(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Unsigned distance from pt to line segment [p1, p2]."""
    seg = p2 - p1
    seg_len_sq = float(seg @ seg)
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(pt - p1))
    t = float(np.clip((pt - p1) @ seg / seg_len_sq, 0.0, 1.0))
    return float(np.linalg.norm(pt - (p1 + t * seg)))


def _signed_dist_to_support(pt_xy: np.ndarray, foot_pts: np.ndarray, pad: float) -> float:
    """Signed distance from pt_xy to the padded support region.

    foot_pts: [N, 2] array of foot contact positions.
    pad:      foot half-length padding (metres).

    Returns
    -------
    negative  → inside the support region  (stable)
    positive  → outside                    (unstable / falling)
    """
    n = len(foot_pts)
    if n == 0:
        return 1.0  # no support at all
    if n == 1:
        return float(np.linalg.norm(pt_xy - foot_pts[0])) - pad
    # 2 points: line segment + padding on both ends
    return _dist_point_to_segment(pt_xy, foot_pts[0], foot_pts[1]) - pad


# --------------------------------------------------------------------------
# Per-PKL labelling
# --------------------------------------------------------------------------

def label_single_pkl(
    input_path: str,
    output_path: str,
) -> tuple[str, str | None, dict[str, float] | None]:
    """Label one PKL.  Returns (input_path, error_or_None, stats_or_None)."""
    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return (input_path, f"load error: {e}", None)

    # ------------------------------------------------------------------
    # Validate required fields
    # ------------------------------------------------------------------
    for field in ("body_pos_w", "body_quat_w", "link_body_list", "fps", "root_pos"):
        if field not in data:
            return (input_path, f"missing field '{field}' — run enrich_pkl first", None)

    body_pos_w: np.ndarray = np.asarray(data["body_pos_w"], dtype=np.float32)  # [T, B, 3]
    link_body_list: list[str] = data["link_body_list"]
    fps: float = float(data["fps"])
    T: int = body_pos_w.shape[0]
    dt: float = 1.0 / fps

    # ------------------------------------------------------------------
    # Build PKL-body → G1-mass lookup
    # ------------------------------------------------------------------
    g1_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(_G1_BODY_NAMES)}
    pkl_to_mass: dict[int, float] = {}
    for pkl_i, name in enumerate(link_body_list):
        if name in g1_name_to_idx:
            pkl_to_mass[pkl_i] = float(_G1_MASSES[g1_name_to_idx[name]])

    # ------------------------------------------------------------------
    # Whole-body CoM  [T, 3]
    # ------------------------------------------------------------------
    com_pos_w = np.zeros((T, 3), dtype=np.float32)
    used_mass = 0.0
    for pkl_i, mass in pkl_to_mass.items():
        com_pos_w += mass * body_pos_w[:, pkl_i, :]
        used_mass += mass
    if used_mass < 1.0:
        return (input_path, f"too little matched mass ({used_mass:.1f} kg)", None)
    com_pos_w /= used_mass

    # ------------------------------------------------------------------
    # CoM velocity  [T, 3]  — central finite differences
    # ------------------------------------------------------------------
    com_vel_w = np.gradient(com_pos_w.astype(np.float64), dt, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Capture point  [T, 2]   CP = CoM_xy + v_xy / ω,  ω = sqrt(g/h)
    # ------------------------------------------------------------------
    h = np.clip(com_pos_w[:, 2], 0.01, None)          # CoM height
    omega = np.sqrt(_G / h)                            # [T]
    cp_xy = com_pos_w[:, :2] + com_vel_w[:, :2] / omega[:, None]  # [T, 2]

    # ------------------------------------------------------------------
    # Foot contact detection from ankle roll link heights
    # ------------------------------------------------------------------
    try:
        left_idx  = link_body_list.index("left_ankle_roll_link")
        right_idx = link_body_list.index("right_ankle_roll_link")
    except ValueError as e:
        return (input_path, f"foot body not found: {e}", None)

    left_contact:  np.ndarray = body_pos_w[:, left_idx,  2] < _FOOT_CONTACT_THRESH  # [T]
    right_contact: np.ndarray = body_pos_w[:, right_idx, 2] < _FOOT_CONTACT_THRESH  # [T]
    foot_contact = np.stack([left_contact, right_contact], axis=1)  # [T, 2]

    left_pos_xy  = body_pos_w[:, left_idx,  :2]  # [T, 2]
    right_pos_xy = body_pos_w[:, right_idx, :2]  # [T, 2]

    # ------------------------------------------------------------------
    # Per-frame stability  (vectorised over frames where possible,
    # but support polygon distance requires a small per-frame loop)
    # ------------------------------------------------------------------
    com_margin_active   = np.empty(T, dtype=np.float32)  # active contacts only
    cp_margin_active    = np.empty(T, dtype=np.float32)
    cp_margin_with_swing = np.empty(T, dtype=np.float32)  # active + projected swing

    for t in range(T):
        # Active contact foot positions
        active_pts: list[np.ndarray] = []
        if left_contact[t]:  active_pts.append(left_pos_xy[t])
        if right_contact[t]: active_pts.append(right_pos_xy[t])

        # Anticipated polygon: add projected swing foot
        anticipated_pts: list[np.ndarray] = list(active_pts)
        if not left_contact[t]:  anticipated_pts.append(left_pos_xy[t])
        if not right_contact[t]: anticipated_pts.append(right_pos_xy[t])

        active_arr     = np.array(active_pts)     if active_pts     else np.empty((0, 2))
        anticipated_arr = np.array(anticipated_pts)

        com_margin_active[t]    = _signed_dist_to_support(com_pos_w[t, :2], active_arr,     _FOOT_PAD)
        cp_margin_active[t]     = _signed_dist_to_support(cp_xy[t],         active_arr,     _FOOT_PAD)
        cp_margin_with_swing[t] = _signed_dist_to_support(cp_xy[t],         anticipated_arr, _FOOT_PAD)

    # ------------------------------------------------------------------
    # Final labels
    # ------------------------------------------------------------------
    # is_statically_stable: CoM AND CP inside active-contact polygon
    is_statically_stable: np.ndarray = (
        (com_margin_active < 0) & (cp_margin_active < 0)
    )

    # is_step_viable: CP inside anticipated polygon (swing foot projected down)
    # CoM check still uses active contacts only — static posture is independent
    is_step_viable: np.ndarray = (
        (com_margin_active < 0) | True  # CoM not required for dynamic walking
    ) & (cp_margin_with_swing < 0)
    # More precisely: CP must be catchable with the next step
    is_step_viable = cp_margin_with_swing < 0

    # stability_margin: most conservative of the two CP margins
    stability_margin: np.ndarray = np.minimum(
        cp_margin_active, cp_margin_with_swing
    ).astype(np.float32)

    # is_falling: CP is far outside even the anticipated polygon
    is_falling: np.ndarray = cp_margin_with_swing > _FALLING_MARGIN

    # ------------------------------------------------------------------
    # Write labelled PKL
    # ------------------------------------------------------------------
    enriched = dict(data)
    enriched["foot_contact"]           = foot_contact.astype(bool)
    enriched["com_pos_w"]              = com_pos_w
    enriched["com_vel_w"]              = com_vel_w
    enriched["capture_point_xy"]       = cp_xy.astype(np.float32)
    enriched["is_statically_stable"]   = is_statically_stable
    enriched["is_step_viable"]         = is_step_viable
    enriched["is_falling"]             = is_falling
    enriched["stability_margin"]       = stability_margin

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(enriched, f)

    # ------------------------------------------------------------------
    # Per-motion summary stats (go into the YAML)
    # ------------------------------------------------------------------
    stats: dict[str, float] = {
        "static_stable_ratio": float(is_statically_stable.mean()),
        "step_viable_ratio":   float(is_step_viable.mean()),
        "falling_ratio":       float(is_falling.mean()),
        "mean_stability_margin": float(stability_margin.mean()),
        "min_stability_margin":  float(stability_margin.min()),
    }
    return (input_path, None, stats)


def _worker(args: tuple[str, str]) -> tuple[str, str | None, dict[str, float] | None]:
    input_path, output_path = args
    return label_single_pkl(input_path, output_path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add per-frame stability labels to enriched PKL dataset"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to enriched dataset YAML (output of enrich_pkl.py)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for labelled PKLs + YAML",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        config = yaml.safe_load(f)

    root_path: str = config["root_path"]
    motions: list[dict[str, Any]] = config["motions"]
    output_dir = Path(args.output_dir)

    # Build work list
    work_items: list[tuple[str, str]] = []
    for entry in motions:
        rel_file = entry["file"]
        input_path  = os.path.join(root_path, rel_file)
        output_path = str(output_dir / rel_file)
        work_items.append((input_path, output_path))

    print(f"Labelling {len(work_items)} PKL files → {output_dir}")

    errors: list[str] = []
    labelled_motions: list[dict[str, Any]] = []

    n_workers = min(args.workers, len(work_items))
    with Pool(processes=n_workers) as pool:
        for input_path, err, stats in tqdm(
            pool.imap_unordered(_worker, work_items, chunksize=16),
            total=len(work_items),
            desc="Labelling",
        ):
            # Find the original entry to preserve any existing metadata
            rel = os.path.relpath(input_path, root_path)
            orig_entry = next(
                (e for e in motions if e["file"] == rel), {"file": rel}
            )
            entry = dict(orig_entry)
            if err is not None:
                errors.append(f"{input_path}: {err}")
            elif stats is not None:
                entry.update(stats)
            labelled_motions.append(entry)

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Sort by step_viable_ratio descending so curriculum sampling is easy
    labelled_motions.sort(
        key=lambda e: e.get("step_viable_ratio", 0.0), reverse=True
    )

    # Write new YAML
    new_config = {
        "root_path": str(output_dir),
        "motions": labelled_motions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = output_dir / "dataset.yaml"
    with open(output_yaml, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    n_ok = len(work_items) - len(errors)
    print(f"\nDone: {n_ok}/{len(work_items)} labelled.")
    print(f"YAML: {output_yaml}")

    # Print dataset-level summary
    if labelled_motions:
        viable = [e.get("step_viable_ratio", 0) for e in labelled_motions]
        static = [e.get("static_stable_ratio", 0) for e in labelled_motions]
        falling = [e.get("falling_ratio", 0) for e in labelled_motions]
        print(f"\nDataset summary ({n_ok} clips):")
        print(f"  step_viable_ratio  — mean {np.mean(viable):.3f}  "
              f"median {np.median(viable):.3f}")
        print(f"  static_stable_ratio— mean {np.mean(static):.3f}  "
              f"median {np.median(static):.3f}")
        print(f"  falling_ratio      — mean {np.mean(falling):.3f}  "
              f"median {np.median(falling):.3f}")
        print(f"\n  Clips with step_viable_ratio > 0.80 (easy): "
              f"{sum(1 for v in viable if v > 0.80)}")
        print(f"  Clips with step_viable_ratio 0.40-0.80 (medium): "
              f"{sum(1 for v in viable if 0.40 <= v <= 0.80)}")
        print(f"  Clips with step_viable_ratio < 0.40 (hard/diverse): "
              f"{sum(1 for v in viable if v < 0.40)}")


if __name__ == "__main__":
    main()
