import pandas as pd
import numpy as np
from itertools import product
from typing import Iterable, Sequence, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from pathlib import Path
from constants.common_functions import lddt_np_dist  # custom lddt function
from tqdm import tqdm


path_1 = Path("output/metrics/glycans/glycoshape_validated_coordinates.pqt")
path_2 = Path("output/metrics/glycans/alphafold_webserver_glycoshape_coordinates.pqt")

label_1 = "GS"
label_2 = "AF_GS"

output_path = Path("output/metrics/glycans/glycoshape_vs_af_glycoshape_lddt.pqt")
group_name = "GlycoShape vs AlphaFold-Webserver Predictions"

def _wrap_progress(it: Iterable, total: Optional[int], desc: str, unit: str, show: bool):
    return tqdm(it, total=total, desc=desc, unit=unit, leave=unit=="subgroup") if show else it

def _fmt_subkey(k: Union[Tuple, str, int]) -> str:
    return " | ".join(map(str, k)) if isinstance(k, tuple) else str(k)

def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ("group_name","g1_label","g2_label","g1_cluster","g2_cluster"):
        if c in df: df[c] = df[c].astype("string")
    if "aligned_atoms" in df: df["aligned_atoms"] = df["aligned_atoms"].astype("Int32")
    if "lddt" in df: df["lddt"] = df["lddt"].astype("float32")
    df["pair_id"] = (
        df["g1_label"] + ":" + df["g1_cluster"] + "|" + df["g2_label"] + ":" + df["g2_cluster"]
    ).astype("string")
    return df


def _coords_by_cluster(df, key_cols=("Chain","Resname","Resid","Atom"), xyz=("X","Y","Z")):
    """
    Group a coordinate DataFrame by 'Cluster' and store:
    - The identifying columns for each atom (keys)
    - The coordinates as an (N, 3) float32 NumPy array
    Returns:
    dict: {cluster_id -> (keys_df, coords_array)}
    """
    out = {}
    for cluster_id, sub_df in df.groupby("Cluster", sort=False):
        # Copy identifying columns so they remain intact
        keys = sub_df.loc[:, key_cols].copy()
        # Extract coordinate columns as a contiguous float32 array
        coords = sub_df.loc[:, xyz].to_numpy(dtype=np.float32, copy=False)
        out[cluster_id] = (keys, coords)
    return out

def _align_coords(entry_a, entry_b):
    """
    Align two clusters so that only atoms present in both are kept,
    and in the same order in both coordinate arrays.

    entry_a: (keys_df, coords_array) for cluster A
    entry_b: (keys_df, coords_array) for cluster B

    Returns:
    coords_a_aligned, coords_b_aligned
    OR (None, None) if no atoms match
    """
    keys_a, coords_a = entry_a
    keys_b, coords_b = entry_b

    # Give each atom in A and B an index so we can recover coords after merging
    keys_a = keys_a.assign(_idx_a=np.arange(len(keys_a)))
    keys_b = keys_b.assign(_idx_b=np.arange(len(keys_b)))

    # Inner join on all key columns (same atom identity)
    merged = keys_a.merge(keys_b, on=list(keys_a.columns[:-1]), how="inner")

    if merged.empty:
        # No matching atoms
        return None, None

    # Get the positions of matching atoms in each array
    ia = merged["_idx_a"].to_numpy()
    ib = merged["_idx_b"].to_numpy()

    # Return coordinates in the same matched order
    return coords_a[ia], coords_b[ib]

def _lddt_from_coords(coords1, coords2, cutoff=15.0, per_residue=False):
    """
    Compute lDDT from two aligned coordinate arrays (no superposition).
    lDDT compares all pairwise distances and checks if they are preserved
    within given tolerances, but only for neighbors within 'cutoff' Å.

    Returns:
    scalar if per_residue=False
    array of per-residue scores if per_residue=True
    """
    L1, L2 = len(coords1), len(coords2)
    if L1 == 0 or L2 == 0 or L1 != L2:
        return np.nan  # invalid input

    # Compute all pairwise distances between atoms in each structure
    d1 = cdist(coords1, coords1)
    d2 = cdist(coords2, coords2)

    # Ignore self-distances (set diagonal to NaN)
    np.fill_diagonal(d1, np.nan)
    np.fill_diagonal(d2, np.nan)
            
    mask = np.ones((L1,1))[None]

    if per_residue:
        return lddt_np_dist(d1, d2, mask, cutoff=15.0, per_residue=True)
    else:
        return lddt_np_dist(d1, d2, mask, cutoff=15.0, per_residue=False)[0]
    
def _load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load Parquet file to DataFrame; only accepts valid file paths."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def calculate_lddt_for_groups(
    group_name: str,
    group_1_path: Union[str, Path],
    group_2_path: Union[str, Path],
    label_1: str = "Group1",
    label_2: str = "Group2",
    avg_vals_dict: Optional[dict] = None,
    key_cols: Sequence[str] = ("Resname","Resid","Atom Name"),
    xyz: Sequence[str] = ("X","Y","Z"),
    cutoff: float = 15.0,
    subgroup_cols: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Concise lDDT calculator between clusters of two tables.
    - Optional subgrouping via `subgroup_cols` (e.g., ["GSID","Configuration"])
    - Stable, tidy schema with labels as values (not headers)
    - Progress bars for subgroups and pairwise comparisons (tqdm)
    """

    group_1 = _load_parquet(group_1_path)
    group_2 = _load_parquet(group_2_path)


    def _run_one(batch_name: str, g1: pd.DataFrame, g2: pd.DataFrame) -> pd.DataFrame:
        # cache clusters once
        g1c = _coords_by_cluster(g1, key_cols=key_cols, xyz=xyz)
        g2c = _coords_by_cluster(g2, key_cols=key_cols, xyz=xyz)

        rows = []
        pairs = product(g1c.keys(), g2c.keys())
        pairs = _wrap_progress(
            pairs, total=len(g1c)*len(g2c), desc=f"Pairs: {batch_name}", unit="pair", show=show_progress
        )

        for c1, c2 in pairs:
            a1, a2 = _align_coords(g1c[c1], g2c[c2])  # aligned coords or (None, None)
            if a1 is None:
                rows.append(dict(group_name=batch_name, g1_label=label_1, g2_label=label_2,
                                 g1_cluster=c1, g2_cluster=c2, aligned_atoms=0, lddt=np.nan))
                continue
            val = _lddt_from_coords(a1, a2, cutoff=cutoff, per_residue=False)
            rows.append(dict(group_name=batch_name, g1_label=label_1, g2_label=label_2,
                             g1_cluster=c1, g2_cluster=c2,
                             aligned_atoms=int(a1.shape[0]),
                             lddt=float(val) if np.isfinite(val) else np.nan))
            if avg_vals_dict is not None:
                avg_vals_dict[(batch_name, c1, c2)] = val

        return _enforce_dtypes(pd.DataFrame(rows))

    # no subgrouping → run once
    if not subgroup_cols:
        return _run_one(group_name, group_1, group_2)

    # subgrouping → intersect keys and run per subgroup
    g1g = group_1.groupby(list(subgroup_cols), dropna=False)
    g2g = group_2.groupby(list(subgroup_cols), dropna=False)
    common = sorted(set(g1g.groups.keys()) & set(g2g.groups.keys()))

    out_frames = []
    sub_iter = _wrap_progress(common, total=len(common), desc=f"Subgroups for {group_name}", unit="subgroup", show=show_progress)
    for key in sub_iter:
        sub_g1, sub_g2 = g1g.get_group(key), g2g.get_group(key)
        sub_name = f"{group_name} | {_fmt_subkey(key)}"
        df = _run_one(sub_name, sub_g1, sub_g2)

        # attach subgroup columns as values
        if isinstance(key, tuple):
            for col, val in zip(subgroup_cols, key):
                df[col] = val
        else:
            df[subgroup_cols[0]] = key
        out_frames.append(df)

    return pd.concat(out_frames, ignore_index=True) if out_frames else _enforce_dtypes(pd.DataFrame(
        columns=["group_name","g1_label","g2_label","g1_cluster","g2_cluster","aligned_atoms","lddt","pair_id", *subgroup_cols]
    ))



df = calculate_lddt_for_groups(group_name, path_1, path_2, label_1, label_2, subgroup_cols=["GSID", "Configuration"])

df.to_parquet(output_path, index=False)  # Save the DataFrame as a parquet file