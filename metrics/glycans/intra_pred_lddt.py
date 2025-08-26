# intra_pred_lddt.py
from __future__ import annotations
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist


# ---------- Helper: group DataFrame by Cluster and store coords ----------

def _coords_by_cluster(df: pd.DataFrame,
                       key_cols: Sequence[str],
                       xyz: Sequence[str]):
    """
    Build a cache of coordinates keyed by cluster ID.

    Parameters:
        df       : DataFrame containing structure info
        key_cols : columns that uniquely identify each atom (e.g., residue + atom name)
        xyz      : coordinate columns (X, Y, Z)

    Returns:
        dict: cluster_id -> (keys_df, coords_array)
              keys_df = atom identifiers for alignment
              coords_array = numpy array (N_atoms, 3) of float32
    """
    out = {}
    for cid, sub in df.groupby("Cluster", sort=False, observed=True):
        keys = sub.loc[:, key_cols].copy()
        coords = sub.loc[:, xyz].to_numpy(dtype=np.float32, copy=False)
        out[cid] = (keys, coords)
    return out


# ---------- Helper: align two clusters by shared atoms ----------

def _align_coords(entry_a, entry_b):
    """
    Align two clusters so they have the same atoms in the same order.

    Parameters:
        entry_a : (keys_df, coords_array) for cluster A
        entry_b : (keys_df, coords_array) for cluster B

    Returns:
        coords_a_aligned, coords_b_aligned : (N_shared_atoms, 3) each
        If no atoms are shared, returns (None, None)
    """
    keys_a, coords_a = entry_a
    keys_b, coords_b = entry_b

    # Assign row indices so we can match positions after merging
    keys_a = keys_a.assign(_ia=np.arange(len(keys_a), dtype=np.int64))
    keys_b = keys_b.assign(_ib=np.arange(len(keys_b), dtype=np.int64))

    # Merge on atom-identity columns (everything except the index helpers)
    merged = keys_a.merge(keys_b, on=list(keys_a.columns[:-1]), how="inner")

    if merged.empty:
        return None, None  # No overlap

    # Get the matched indices for both coordinate arrays
    ia = merged["_ia"].to_numpy()
    ib = merged["_ib"].to_numpy()

    return coords_a[ia], coords_b[ib]


# ---------- Helper: compute scalar lDDT ----------

def _lddt_from_coords(coords1: np.ndarray,
                      coords2: np.ndarray,
                      cutoff: float = 15.0) -> float:
    """
    Compute scalar lDDT between two aligned coordinate arrays.

    Parameters:
        coords1, coords2 : numpy arrays of shape (N_atoms, 3)
        cutoff            : distance cutoff for neighbor definition

    Returns:
        lDDT value (float) or NaN if invalid
    """
    L = len(coords1)
    if L == 0 or L != len(coords2):
        return float("nan")

    # Pairwise distances within each structure
    d1 = cdist(coords1, coords1)
    d2 = cdist(coords2, coords2)

    # Ignore self-distances
    np.fill_diagonal(d1, np.nan)
    np.fill_diagonal(d2, np.nan)

    # Absolute differences between distances
    diff = np.abs(d1 - d2)

    # Only consider neighbors within the cutoff distance
    neighbor_mask = d1 <= cutoff

    # Fraction of neighbors within tolerance for each tolerance threshold
    tolerances = (0.5, 1.0, 2.0, 4.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_tol = [((diff <= t) & neighbor_mask).sum(1) / neighbor_mask.sum(1) for t in tolerances]
        per_res = np.nanmean(np.stack(per_tol, axis=1), axis=1)

    # Mean over residues → scalar lDDT
    return float(np.nanmean(per_res))


# ---------- Main function: compute all intra-group lDDT pairs ----------

def lddt_within_group(
    coords_path: Union[str, Path],
    label: str = "Pred",
    group_name: Optional[str] = None,
    key_cols: Sequence[str] = ("Resname", "Resid", "Atom Name"),
    xyz: Sequence[str] = ("X", "Y", "Z"),
    cutoff: float = 15.0,
    subgroup_cols: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute lDDT for all unordered cluster pairs within a single group.

    Parameters:
        coords_path   : path to a Parquet file containing coordinates
        label         : label for this group (e.g., "AF", "GS")
        group_name    : descriptive name for output; defaults to filename stem
        key_cols      : atom-identity columns
        xyz           : coordinate columns
        cutoff        : lDDT neighbor cutoff distance
        subgroup_cols : columns to split the data on before computing
        show_progress : whether to show tqdm progress bars

    Returns:
        DataFrame with columns:
            group_name, g1_label, g2_label, g1_cluster, g2_cluster,
            aligned_atoms, lddt, pair_id
        Plus subgroup columns if specified
    """
    coords_path = Path(coords_path)
    if not coords_path.exists():
        raise FileNotFoundError(coords_path)

    df = pd.read_parquet(coords_path)
    base_name = group_name or coords_path.stem

    if subgroup_cols:
        # Process each subgroup separately and concatenate results
        out_frames = []
        groups = df.groupby(list(subgroup_cols), dropna=False, observed=True)
        keys = sorted(groups.groups.keys())

        subgroup_iter = keys
        if show_progress:
            subgroup_iter = tqdm(subgroup_iter, total=len(keys),
                                 desc=f"Subgroups: {base_name}", unit="subgroup")

        for key in subgroup_iter:
            sub_df = groups.get_group(key)
            subgroup_name = f"{base_name} | " + " | ".join(map(str, key if isinstance(key, tuple) else (key,)))
            out = _within_core(sub_df, subgroup_name, label, key_cols, xyz, cutoff, show_progress)

            # Attach subgroup column values
            if isinstance(key, tuple):
                for col, val in zip(subgroup_cols, key):
                    out[col] = val
            else:
                out[subgroup_cols[0]] = key

            out_frames.append(out)

        if not out_frames:
            return _empty_frame(subgroup_cols)
        return _enforce_dtypes(pd.concat(out_frames, ignore_index=True))

    # No subgrouping → single run
    return _enforce_dtypes(_within_core(df, base_name, label, key_cols, xyz, cutoff, show_progress))


# ---------- Core routine for one subgroup ----------

def _within_core(df: pd.DataFrame,
                 group_name: str,
                 label: str,
                 key_cols: Sequence[str],
                 xyz: Sequence[str],
                 cutoff: float,
                 show_progress: bool) -> pd.DataFrame:
    """
    Compute lDDT for all unordered cluster pairs in a DataFrame.
    """
    cache = _coords_by_cluster(df, key_cols=key_cols, xyz=xyz)
    clusters = list(cache.keys())

    if len(clusters) < 2:
        return _empty_frame()

    rows = []
    pairs = combinations(clusters, 2)  # (A,B) with A < B
    if show_progress:
        pairs = tqdm(pairs, total=len(clusters) * (len(clusters) - 1) // 2,
                     desc=f"Pairs (within): {group_name}", unit="pair", leave=False)

    for c1, c2 in pairs:
        # Stringify cluster IDs for consistency
        s1, s2 = str(c1), str(c2)
        if s2 < s1:  # enforce ordering
            c1, c2, s1, s2 = c2, c1, s2, s1

        coords1, coords2 = _align_coords(cache[c1], cache[c2])
        if coords1 is None:
            rows.append(dict(group_name=group_name, g1_label=label, g2_label=label,
                             g1_cluster=s1, g2_cluster=s2,
                             aligned_atoms=0, lddt=np.nan))
            continue

        lddt_val = _lddt_from_coords(coords1, coords2, cutoff=cutoff)
        rows.append(dict(group_name=group_name, g1_label=label, g2_label=label,
                         g1_cluster=s1, g2_cluster=s2,
                         aligned_atoms=int(coords1.shape[0]),
                         lddt=float(lddt_val) if np.isfinite(lddt_val) else np.nan))

    return pd.DataFrame(rows)


# ---------- Helpers for output formatting ----------

def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce consistent dtypes for output DataFrame."""
    if df.empty:
        return df
    for col in ("group_name", "g1_label", "g2_label", "g1_cluster", "g2_cluster"):
        if col in df:
            df[col] = df[col].astype("string")
    if "aligned_atoms" in df:
        df["aligned_atoms"] = df["aligned_atoms"].astype("Int32")
    if "lddt" in df:
        df["lddt"] = df["lddt"].astype("float32")

    # Add unique pair ID
    df["pair_id"] = (
        df["g1_label"] + ":" + df["g1_cluster"] + "|" + df["g2_label"] + ":" + df["g2_cluster"]
    ).astype("string")
    return df

def _empty_frame(extra_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Create an empty results DataFrame with the correct schema."""
    base_cols = ["group_name", "g1_label", "g2_label", "g1_cluster", "g2_cluster",
                 "aligned_atoms", "lddt", "pair_id"]
    if extra_cols:
        base_cols += list(extra_cols)
    return pd.DataFrame(columns=base_cols)
