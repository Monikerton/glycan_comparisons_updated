# ingest/common.py

import os
import sys
from pathlib import Path
from collections import defaultdict
import glycosylator as gl
from glycosylator import resources
import numpy as np
import pandas as pd

def glycoshape_ROH_corrector(path_in: Path, path_out: Path) -> None:
    """
    GlycoShape PDB files contain an ROH residue as the first residue in every file.
    Combine that ROH residue with the next residue and write a new PDB file.
    """
    with open(path_in, "r") as f:
        lines = [x for x in f.readlines() if x.startswith("HETATM") or x.startswith("TER") or x.startswith("END")]
        i = 0
        # Skip headers until first non-ROH residue
        while i < len(lines) and "ROH" in lines[i]:
            i += 1
        first_res = lines[i][17:20]
        first_res_num = lines[i][25:26]

    os.makedirs(path_out.parent, exist_ok=True)
    with open(path_out, "w") as f:
        for line in lines:
            if "ROH" in line:
                line = line.replace("ROH", first_res)
                line = line[:25] + first_res_num + line[26:]
            f.write(line)


def get_components(file_path: Path) -> dict:
    """
    Parse a GlycoShape structure file and return a dict of component counts.
    """
    path_str = str(file_path)
    if path_str.endswith(".cif"):
        g = gl.Glycan.from_cif(path_str)
    elif path_str.endswith(".pdb"):
        g = gl.Glycan.from_pdb(path_str)
    else:
        raise AssertionError(f"Uncertain which filetype this is: {path_str}")

    g.infer_bonds(max_bond_length=1.622, restrict_residues=True)
    g.infer_residue_connections(bond_length=1.62)
    g.infer_glycan_tree()

    hist = g.hist().set_index("residue").to_dict()["count"]
    components = {}
    for k, v in hist.items():
        key = k.split("-")[-1]
        components[key] = components.get(key, 0) + v

    return components


def get_tree(glycan, replace_root: bool = False) -> tuple:
    """
    Traverse glycan tree to generate a string representation and count of residues.
    """
    residues = list(glycan.get_residues())
    if len(residues) == 1:
        return residues[0].resname, 1

    segments = glycan._glycan_tree._segments
    tree = {}

    for segment in segments:
        a, b = segment
        if a.serial_number > b.serial_number:
            a, b = b, a
        tree.setdefault(a, []).append(b)

    root = segments[0][0]
    
    return traverse(tree, root, 0, 0)

def traverse(tree, root, d, num, replace_root=False):
    if root not in tree:
        print("Reached leaf", root)
        return root.resname, num + 1
    else:
        children = tree[root]
        s = root.resname
        print("Node", root, "has children", children)
        if d == 0 and replace_root:
            s = s.replace("NDG", "NAG")

        for child in children:
            t, num = traverse(tree, child, d+1, num)
            s += "(" + t + ")"
        return s, num + 1


def get_angles(structure, ref_residues: set, verbose: bool = False) -> dict:
    """
    Compute glycan dihedral angles (phi, psi, omega) for each bond.
    """
    data = defaultdict(lambda: defaultdict(list))
    connections = structure.get_residue_connections(triplet=False)
    residue_connections = [(i, i.parent, j, j.parent) for i, j in connections]

    for atom1, res1, atom2, res2 in residue_connections:
        if res1.resname not in ref_residues or res2.resname not in ref_residues:
            if verbose:
                print(f"Skipping pair {res1.resname}-{res2.resname}")
            continue
        if verbose:
            print(f"Computing angles for {res1.resname} {res1.id[1]} -> {res2.resname} {res2.id[1]}")
        phi, psi, omega = get_groups(structure, atom1, atom2)
        key = (f"{res1.resname} {res1.id[1]}", f"{res2.resname} {res2.id[1]}")
        data[key]["phi"].append(phi)
        data[key]["psi"].append(psi)
        data[key]["omega"].append(omega)

    return data


def get_ref_residues() -> set:
    """
    Load reference glycan residue IDs from glycosylator and add extras.
    """
    ref_residues = resources.reference_glycan_residue_ids()
    # Add known extras
    extras = {"4YB", "0MB", "3lB", "2lA", "0FA", "3VA"}
    return ref_residues | extras


def get_groups(structure, atom1, atom2):
    """
    Identify and compute the glycosidic dihedral angles (phi, psi, omega)
    between two connected residues/atoms.
    """
    # Ensure atom1 is oxygen, atom2 is carbon
    if "O" in atom1.id and "C" in atom2.id:
        pass
    elif "O" in atom2.id and "C" in atom1.id:
        atom1, atom2 = atom2, atom1
    else:
        raise ValueError("No valid O-C atom pair found for dihedral calculation")

    # Neighbors for phi and psi
    n1 = list(structure.get_neighbors(atom1, 1))
    n1.remove(atom2)
    n2 = [x for x in structure.get_neighbors(atom2, 1) if "O" in x.id]
    n2.remove(atom1)
    assert len(n1) == 1, "Unexpected neighbor count for atom1"

    # Second neighbor for psi
    n1_n = [x for x in structure.get_neighbors(n1[0], 1) if "C" in x.id and x != atom1]
    n1_n = sorted(n1_n, key=lambda x: x.id)[0]

    # Define dihedral atom groups
    g1 = (n1[0], atom1, atom2, n2[0])  # phi
    g2 = (n1_n, n1[0], atom1, atom2)  # psi
    phi = structure.compute_dihedral(*g1)
    psi = structure.compute_dihedral(*g2)

    # Omega for 1-6 linkages
    omega = np.nan
    if n1[0].id == "C6":
        n1_n_n = [x for x in structure.get_neighbors(n1_n, 1) if "O" in x.id]
        assert len(n1_n_n) == 1
        g3 = (atom1, n1[0], n1_n, n1_n_n[0])
        omega = structure.compute_dihedral(*g3)

    return phi, psi, omega

def get_error_message(e: Exception) -> None:
    """
    Print detailed error context: exception type and source line.
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    if exc_tb is not None:
        line_number = exc_tb.tb_lineno
        filename = exc_tb.tb_frame.f_code.co_filename
        try:
            with open(filename) as f:
                error_line = f.readlines()[line_number - 1].strip()
            print(f"{exc_type.__name__} on line {line_number} of {filename}: {error_line}")
        except Exception:
            print(f"{exc_type.__name__}: {e}")
    else:
        print(f"{exc_type.__name__}: {e}")
        
from pymol import cmd

def get_pymol_residue_ids(obj_name: str, gs_id: str):
    """
    Get unique (resi, resn, gs_id) tuples from a loaded PyMOL object.
    """
    # Efficiently pull out only the columns you need
    model = cmd.get_model(obj_name)
    # Use a set comprehension to avoid the explicit loop
    residues = {
        (atom.resi, atom.resn, gs_id)
        for atom in model.atom
    }
    # Sort by numeric residue ID if possible:
    try:
        return sorted(residues, key=lambda x: int(x[0]))
    except ValueError:
        return sorted(residues)

CODE_MAP = {
    "A": ("alpha", "pyranose"),
    "B": ("beta", "pyranose"),
    "D": ("alpha", "furanose"),
    "U": ("beta", "furanose"),
}

def convert_glycam_to_pdb(glycam_code: str, chart: pd.DataFrame) -> str:
    """
    Convert a 3-letter Glycam code to the corresponding PDB code.
    Returns the PDB code or None if unmapped (excluding ROH).
    """
    if glycam_code == "ROH":
        return "ROH"

    # Precomputed mapping from glycam_code -> (second, third char) -> column
    # Build this mapping once at import‐time, not per‐call
    # e.g. MAPPING = {"A": ("alpha", "D", "pyranose"), ...}
    try:
        second, third = glycam_code[1].upper(), glycam_code[2]
    except IndexError:
        return None

    # Determine ring/config/enantiomer via lookup tables
    mapping = CODE_MAP.get(third)
    if not mapping:
        return None
    config, ring_type = mapping  # e.g. ("alpha", "pyranose")

    enantiomer = "D" if glycam_code[1].isupper() else "L"
    col = f"{config}-{enantiomer}-{ring_type}"

    # Fast .at lookup and catch missing entries
    try:
        return chart.at[second, col]
    except KeyError:
        return None

def create_structure_replace_glycam_code_with_pdb_code(
    input_file: Path,
    output_file: Path,
    chart: pd.DataFrame
) -> None:
    """
    Read input PDB, replace residue names using `chart`, and write.
    """
    with input_file.open() as inf, output_file.open("w") as outf:
        for line in inf:
            if line.startswith(("ATOM  ", "HETATM", "TER   ")):
                resn = line[17:20].strip()
                pdb_code = convert_glycam_to_pdb(resn, chart) or resn
                # Build the new line more directly
                outf.write(f"{line[:17]}{pdb_code:<3}{line[20:]}")
            else:
                outf.write(line)


        
