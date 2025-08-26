# eval_metrics.py
from __future__ import annotations
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist


# --------------------- Precision / Recall (from pred↔crystal table) ---------------------

def compute_precision_recall(pairs_df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute precision/recall from a cross-table of lDDT between predictions and crystals.

    Expected columns in pairs_df:
      - 'g1_cluster' : prediction cluster ID
      - 'g2_cluster' : crystal cluster ID (or structure ID)
      - 'lddt'       : lDDT score for that pair (float, NaN allowed)

    Returns:
      (precision, recall)
        precision = mean over predictions of max lDDT to any crystal
        recall    = mean over crystals    of max lDDT to any prediction
    """
    df = pairs_df.dropna(subset=["lddt"])
    if df.empty:
        return float("nan"), float("nan")

    # For each prediction, take its best match to any crystal
    best_per_pred = df.groupby("g1_cluster", observed=True)["lddt"].max()
    precision = float(best_per_pred.mean()) if len(best_per_pred) else float("nan")

    # For each crystal, take its best match to any prediction
    best_per_cryst = df.groupby("g2_cluster", observed=True)["lddt"].max()
    recall = float(best_per_cryst.mean()) if len(best_per_cryst) else float("nan")

    return precision, recall


# --------------------- Diversity (from precomputed pred↔pred lDDT pairs) ---------------------

def compute_diversity_from_pairs(intra_pairs_df: pd.DataFrame) -> float:
    """
    Compute diversity as mean(1 - lDDT) over unordered prediction–prediction pairs.

    Expected columns in intra_pairs_df:
      - 'g1_cluster', 'g2_cluster' : prediction IDs for each pair
      - 'lddt'                      : lDDT for that pair

    Notes:
      - Self-pairs are ignored.
      - If both (A,B) and (B,A) exist, we only keep one (unordered).
    """
    d = intra_pairs_df.dropna(subset=["lddt"]).copy()
    if d.empty:
        return float("nan")

    # Keep only unordered, non-diagonal pairs
    # Works when cluster IDs are strings; if not, cast to string first.
    mask = d["g1_cluster"] < d["g2_cluster"]
    d = d[mask]
    if d.empty:
        return float("nan")

    return float((1.0 - d["lddt"]).mean())


# --------------------- Diversity (recompute from prediction coordinates) ---------------------

def _coords_by_cluster(df: pd.DataFrame,
                       key_cols: Sequence[str] = ("Resname", "Resid", "Atom Name"),
                       xyz: Sequence[str] = ("X", "Y", "Z")) -> dict:
    """
    Build a cache: cluster_id -> (keys_df, coords_array).
    keys_df contains atom identity columns; coords_array is (N,3) float32.
    """
    out = {}
    for cid, sub in df.groupby("Cluster", sort=False, observed=True):
        keys = sub.loc[:, key_cols].copy()
        coords = sub.loc[:, xyz].to_numpy(dtype=np.float32, copy=False)
        out[cid] = (keys, coords)
    return out

def _align_coords(entry_a: Tuple[pd.DataFrame, np.ndarray],
                  entry_b: Tuple[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Align two clusters so they have the same atoms, in the same order.
    Returns two (N,3) arrays or (None, None) if no overlap.
    """
    keys_a, coords_a = entry_a
    keys_b, coords_b = entry_b
    keys_a = keys_a.assign(_ia=np.arange(len(keys_a), dtype=np.int64))
    keys_b = keys_b.assign(_ib=np.arange(len(keys_b), dtype=np.int64))
    merged = keys_a.merge(keys_b, on=list(keys_a.columns[:-1]), how="inner")
    if merged.empty:
        return None, None
    ia = merged["_ia"].to_numpy()
    ib = merged["_ib"].to_numpy()
    return coords_a[ia], coords_b[ib]

def _lddt_from_coords(coords1: np.ndarray,
                      coords2: np.ndarray,
                      cutoff: float = 15.0) -> float:
    """
    Scalar lDDT (superposition-free) using Cα-like neighborhood with tolerances {0.5,1,2,4} Å.
    Returns NaN if lengths differ or are zero.
    """
    L = len(coords1)
    if L == 0 or L != len(coords2):
        return float("nan")

    d1 = cdist(coords1, coords1)
    d2 = cdist(coords2, coords2)
    np.fill_diagonal(d1, np.nan)
    np.fill_diagonal(d2, np.nan)

    diff = np.abs(d1 - d2)
    neighbor_mask = d1 <= cutoff
    tolerances = (0.5, 1.0, 2.0, 4.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        per_tol = [((diff <= t) & neighbor_mask).sum(1) / neighbor_mask.sum(1) for t in tolerances]
        per_res = np.nanmean(np.stack(per_tol, axis=1), axis=1)

    return float(np.nanmean(per_res))

def compute_diversity_from_coords(pred_coords_path: Path | str,
                                  key_cols: Sequence[str] = ("Resname", "Resid", "Atom Name"),
                                  xyz: Sequence[str] = ("X", "Y", "Z"),
                                  cutoff: float = 15.0,
                                  show_progress: bool = True) -> float:
    """
    Compute diversity directly from a prediction coordinate file (Parquet).
    Diversity = mean(1 - lDDT) over all unordered pairs of prediction clusters.
    """
    df = pd.read_parquet(pred_coords_path)
    cache = _coords_by_cluster(df, key_cols=key_cols, xyz=xyz)
    clusters = list(cache.keys())
    if len(clusters) < 2:
        return float("nan")

    vals: list[float] = []
    it = combinations(clusters, 2)
    if show_progress:
        it = tqdm(it, total=len(clusters) * (len(clusters) - 1) // 2,
                  desc="Computing intra-pred diversity", unit="pair")

    for c1, c2 in it:
        a1, a2 = _align_coords(cache[c1], cache[c2])
        if a1 is None:
            continue
        l = _lddt_from_coords(a1, a2, cutoff=cutoff)
        if np.isfinite(l):
            vals.append(1.0 - l)

    return float(np.mean(vals)) if vals else float("nan")


# --------------------- One-call wrapper to produce a summary row ---------------------

def summarize_metrics(pairs_df: pd.DataFrame,
                      diversity_mode: str = "pairs",
                      diversity_pairs_df: pd.DataFrame | None = None,
                      pred_coords_path: Path | str | None = None,
                      cutoff: float = 15.0) -> pd.DataFrame:
    """
    Produce a one-row DataFrame with precision/recall/diversity.

    diversity_mode:
      - "pairs":   use `diversity_pairs_df` (intra-pred lDDT table)
      - "coords":  recompute diversity from coordinates file `pred_coords_path`
    """
    precision, recall = compute_precision_recall(pairs_df)

    if diversity_mode == "pairs":
        diversity = compute_diversity_from_pairs(diversity_pairs_df) if diversity_pairs_df is not None else float("nan")
        src = "pairs"
    elif diversity_mode == "coords":
        diversity = compute_diversity_from_coords(pred_coords_path, cutoff=cutoff) if pred_coords_path is not None else float("nan")
        src = "coords"
    else:
        raise ValueError("diversity_mode must be 'pairs' or 'coords'")

    return pd.DataFrame([{
        "precision": precision,
        "recall": recall,
        "diversity": diversity,
        "diversity_mode": src,
        "cutoff": cutoff
    }])

def save_summary_metrics(df: pd.DataFrame, output_path: Path | str) -> None:
    """
    Save the summary metrics DataFrame to a parquet file.
    df: DataFrame containing summary metrics
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    df.to_parquet(str(output_path), index=False)  # Save DataFrame as a parquet file
    print(f"Summary metrics saved to {output_path}")