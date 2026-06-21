"""Generate a curriculum-weighted YAML from a stability-labelled dataset.

Reads the output of label_stability_pkl.py (which has per-motion stability
stats) and writes a new YAML with a ``weight`` field added to every motion.

Weighting strategy
------------------
weight = max(min_weight, step_viable_ratio)

This gives easy motions (ratio ≈ 1.0) a full sampling weight and hard /
falling motions (ratio ≈ 0.0) a floor weight so they are still occasionally
sampled, preventing the policy from never seeing difficult transitions.

Usage
-----
    uv run python -m twist2_mjlab.scripts.make_curriculum_yaml \\
        --input  data/labelled/dataset.yaml \\
        --output data/labelled/curriculum_dataset.yaml \\
        --min-weight 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add curriculum weights to a stability-labelled dataset YAML"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to labelled dataset YAML (output of label_stability_pkl.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the curriculum-weighted YAML",
    )
    parser.add_argument(
        "--min-weight", type=float, default=0.05,
        help="Floor weight for hard/falling motions (default: 0.05)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        config = yaml.safe_load(f)

    motions = config["motions"]
    for entry in motions:
        ratio = float(entry.get("step_viable_ratio", 1.0))
        entry["weight"] = max(args.min_weight, ratio)

    weights = [e["weight"] for e in motions]
    print(f"Loaded {len(motions)} motions.")
    print(f"  weight range:  [{min(weights):.3f}, {max(weights):.3f}]")
    print(f"  weight mean:   {np.mean(weights):.3f}")
    print(f"  weight median: {np.median(weights):.3f}")

    # Count effective curriculum tiers
    easy   = sum(1 for w in weights if w >= 0.80)
    medium = sum(1 for w in weights if 0.40 <= w < 0.80)
    hard   = sum(1 for w in weights if w <  0.40)
    total_weight = sum(weights)
    easy_share   = sum(w for w in weights if w >= 0.80)  / total_weight
    medium_share = sum(w for w in weights if 0.40 <= w < 0.80) / total_weight
    hard_share   = sum(w for w in weights if w <  0.40)  / total_weight
    print(f"\nCurriculum tiers (by weight share):")
    print(f"  Easy   (w≥0.80): {easy:6d} clips  ({100*easy_share:.1f}% of sampling mass)")
    print(f"  Medium (0.40-0.80): {medium:6d} clips  ({100*medium_share:.1f}% of sampling mass)")
    print(f"  Hard   (w<0.40): {hard:6d} clips  ({100*hard_share:.1f}% of sampling mass)")

    out_config = {
        "root_path": config["root_path"],
        "motions": motions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(out_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
