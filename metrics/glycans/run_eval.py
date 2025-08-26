#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from pairwise_lddt import calculate_lddt_for_groups
from intra_pred_lddt import lddt_within_group
from metrics.eval_metrics import compute_precision_recall, compute_diversity_from_pairs


def parse_args() -> argparse.Namespace:
    """
    Minimal CLI:
      python run_eval.py path/to/config.yaml
    Optional quick overrides:
      --outdir results2  --cutoff 12.0  --no-progress
    """
    p = argparse.ArgumentParser(description="Evaluate lDDT metrics from YAML config.")
    p.add_argument("config", nargs="?",  # makes it optional
            default="configs/metrics/run_glycan_metrics.yaml", help="Path to YAML config file (default: config.yaml)")   
    p.add_argument("--outdir", type=str, help="Override output directory from YAML.")
    p.add_argument("--cutoff", type=float, help="Override lDDT cutoff from YAML.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return p.parse_args()


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"[error] Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def require(cfg: dict, keys: list[str]) -> None:
    """Fail fast if required keys are missing."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        sys.exit(f"[error] Missing required config keys: {missing}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ---- Required keys in YAML ----
    require(cfg, ["group_name", "pred_file", "cryst_file", "outdir"])

    # ---- Optional keys with defaults ----
    cfg.setdefault("label_1", "Pred")
    cfg.setdefault("label_2", "Cryst")
    cfg.setdefault("key_cols", ["Resname", "Resid", "Atom Name"])
    cfg.setdefault("xyz", ["X", "Y", "Z"])
    cfg.setdefault("cutoff", 15.0)
    cfg.setdefault("subgroup_cols", None)
    cfg.setdefault("show_progress", True)
    cfg.setdefault("also_diversity", False)

    # ---- CLI overrides (take precedence over YAML) ----
    if args.outdir:
        cfg["outdir"] = args.outdir
    if args.cutoff is not None:
        cfg["cutoff"] = args.cutoff
    if args.no_progress:
        cfg["show_progress"] = False

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1) pred ↔ crystal lDDT (pairs table) ----
    pairs = calculate_lddt_for_groups(
        group_name=cfg["group_name"],
        group_1_path=cfg["pred_file"],
        group_2_path=cfg["cryst_file"],
        label_1=cfg["label_1"],
        label_2=cfg["label_2"],
        key_cols=tuple(cfg["key_cols"]),
        xyz=tuple(cfg["xyz"]),
        cutoff=cfg["cutoff"],
        subgroup_cols=cfg["subgroup_cols"],
        show_progress=cfg["show_progress"],
    )
    p_pairs = outdir / "pred_cryst_pairs.pqt"
    pairs.to_parquet(p_pairs, index=False)

    # ---- 2) Precision & Recall ----
    precision, recall = compute_precision_recall(pairs)

    # ---- 3) Diversity (optional) ----
    diversity = float("nan")
    if cfg["also_diversity"]:
        pred_pairs = lddt_within_group(
            coords_path=cfg["pred_file"],
            label=cfg["label_1"],
            key_cols=tuple(cfg["key_cols"]),
            xyz=tuple(cfg["xyz"]),
            cutoff=cfg["cutoff"],
            show_progress=cfg["show_progress"],
        )
        p_predpairs = outdir / "pred_pred_pairs.pqt"
        pred_pairs.to_parquet(p_predpairs, index=False)
        diversity = compute_diversity_from_pairs(pred_pairs)

    # ---- 4) Save summary ----
    summary = pd.DataFrame(
        [
            {
                "precision": precision,
                "recall": recall,
                "diversity": diversity,
                "pairs_file": str(p_pairs.resolve()),
                "pred_file": str(Path(cfg["pred_file"]).resolve()),
                "cryst_file": str(Path(cfg["cryst_file"]).resolve()),
                "cutoff": cfg["cutoff"],
                "computed_diversity": cfg["also_diversity"],
            }
        ]
    )
    summary_path = outdir / "summary.metrics.pqt"
    summary.to_parquet(summary_path, index=False)

    print(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | Diversity: {diversity:.4f}\n"
        f"Saved pairs → {p_pairs}\nSaved summary → {summary_path}"
        + (f"\nSaved intra-pred pairs → {p_predpairs}" if cfg["also_diversity"] else "")
    )


if __name__ == "__main__":
    main()
