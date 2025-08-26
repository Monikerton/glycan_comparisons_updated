"""
This script validates GlycoShape JSON summaries within each GSxxxxx folder and, if valid,
copies **all** corresponding alpha/beta PDBs into a flattened destination.

Validation:
  - All `clusterN_alpha_summary.json` in a GS folder must be identical to each other.
  - All `clusterN_beta_summary.json`  in a GS folder must be identical to each other.
  - If either alpha or beta summaries disagree, the entire GS folder is skipped.

Copying:
  - For GS folders that pass, copy every `clusterN_alpha.pdb` and `clusterN_beta.pdb`
    to `dst` as: `GSID_alpha_clusterN.pdb` and `GSID_beta_clusterN.pdb`.

Usage:
  - Set `src` to the root containing GSxxxxx subfolders.
  - Set `dst` to your processed/validated location.
  - Run directly with `python this_script.py`.
"""

from pathlib import Path
import json, shutil

# --- configure here ---
src = Path("output/ingest/glycoshape/glycoshape_pdbs")
dst = Path("output/processed/glycoshape_pdbs_validated")  # 
# ----------------------

dst.mkdir(parents=True, exist_ok=True)

def load_summaries(gs_dir: Path, kind: str):
    """clusterN_<kind>_summary.json -> {cluster_id:int -> dict}"""
    out = {}
    for p in gs_dir.glob(f"cluster*_{kind}_summary.json"):
        try:
            cid = int(p.stem.split("_")[0].removeprefix("cluster"))
        except ValueError:
            continue
        out[cid] = json.loads(p.read_text())
    return out

def jdiff(a, b, path=""):
    """Return list of differing key paths between two JSON-like objects."""
    if type(a) != type(b):
        return [path or "<root> (type mismatch)"]
    if isinstance(a, dict):
        diffs = []
        akeys, bkeys = set(a), set(b)
        for k in sorted(akeys - bkeys): diffs.append(f"{path+'.' if path else ''}{k}")
        for k in sorted(bkeys - akeys): diffs.append(f"{path+'.' if path else ''}{k}")
        for k in sorted(akeys & bkeys):
            diffs += jdiff(a[k], b[k], f"{path+'.' if path else ''}{k}")
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path} (list length {len(a)} != {len(b)})"]
        diffs = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            diffs += jdiff(ai, bi, f"{path}[{i}]")
        return diffs
    return ([] if a == b else [path or "<root>"])

def check_equal(summaries: dict[int, dict]):
    """True/False, baseline_cluster, mismatches{cid:[paths]}"""
    if len(summaries) <= 1:
        return True, None, {}
    base_id = min(summaries)
    base = summaries[base_id]
    mismatches = {cid: jdiff(base, s)
                  for cid, s in summaries.items()
                  if cid != base_id and s != base}
    return (len(mismatches) == 0), base_id, mismatches

def list_cluster_pdbs(gs_dir: Path, kind: str):
    """Yield (cluster_id:int, path) for all clusterN_<kind>.pdb present."""
    for p in sorted(gs_dir.glob(f"cluster*_{kind}.pdb")):
        try:
            cid = int(p.stem.split("_")[0].removeprefix("cluster"))
        except ValueError:
            continue
        yield cid, p

# Stats
scanned = copied_dirs = skipped_mismatch = 0
copied_files = 0

for gs_dir in sorted(p for p in src.iterdir() if p.is_dir()):
    scanned += 1
    gs_id = gs_dir.name

    # Load and validate summaries
    alpha_s = load_summaries(gs_dir, "alpha")
    beta_s  = load_summaries(gs_dir, "beta")
    a_ok, a_base, a_mm = check_equal(alpha_s)
    b_ok, b_base, b_mm = check_equal(beta_s)

    if not (a_ok and b_ok):
        skipped_mismatch += 1
        print(f"[SKIP] {gs_id} due to summary mismatch:")
        if not a_ok:
            print("  ALPHA diffs:")
            for cid, diffs in a_mm.items():
                print(f"    vs cluster{cid}: " + ", ".join(diffs[:5]) +
                      (f" … +{len(diffs)-5}" if len(diffs) > 5 else ""))
        if not b_ok:
            print("  BETA diffs:")
            for cid, diffs in b_mm.items():
                print(f"    vs cluster{cid}: " + ", ".join(diffs[:5]) +
                      (f" … +{len(diffs)-5}" if len(diffs) > 5 else ""))
        continue

    # Passed: copy ALL cluster PDBs for alpha and beta
    did_copy = False
    for kind in ("alpha", "beta"):
        # Copy only PDBs whose cluster also has a summary (keeps things consistent)
        summary_cids = set((alpha_s if kind == "alpha" else beta_s).keys())
        any_kind = False
        for cid, pdb_path in list_cluster_pdbs(gs_dir, kind):
            if cid not in summary_cids:
                # PDB with no corresponding summary: optional warning
                print(f"[warn] {gs_id}: {pdb_path.name} has no matching {kind} summary; skipping")
                continue
            out = dst / f"{gs_id}_{kind}_cluster{cid}.pdb"
            shutil.copy2(pdb_path, out)
            copied_files += 1
            any_kind = True
            print(f"[COPY] {pdb_path} -> {out}")
        did_copy = did_copy or any_kind

    if did_copy:
        copied_dirs += 1

print("\nDone.")
print(f"Scanned GS folders: {scanned}")
print(f"Copied from: {copied_dirs} folders")
print(f"PDB files copied: {copied_files}")
print(f"Skipped (any mismatch): {skipped_mismatch}")
