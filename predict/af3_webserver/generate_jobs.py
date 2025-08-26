import os
import json
import yaml
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import glycosylator as gl
import re
import copy
import itertools
import string
from constants.paths import PREPROCESS_DIRECTORY, OUTPUT_DIRECTORY, RAW_DATA_DIRECTORY

LONGEST_GLYCAN_RESIDUE = man_6 = "MAN(MAN(MAN(MAN(MAN(MAN))))" # long glycan residue for generating longest glycan sequence, mannose 6
"""
generate_jobs.py --predict_config <predict_config.yaml> --cif_folder <cif_folder_name> (under preprocess/ and raw_data)

"""


# ----------------------------------------
# YOUR EXISTING FUNCTIONS
# ----------------------------------------

def get_tree(glycan, replace_root=False):
    residues = list(glycan.get_residues())
    
    if len(residues) == 1:
        return residues[0].resname, 1

    segments = glycan._glycan_tree._segments

    tree = {}
    for segment in segments:
        a, b = segment
        if a.serial_number > b.serial_number:
            a, b = b, a

        if a in tree:
            tree[a].append(b)
        else:
            tree[a] = [b]

    root = glycan.get_root().get_parent()

    return traverse(tree, root, d=0, num=0, replace_root=replace_root)

def traverse(tree, node, d=0, num=0, replace_root=False):
    """
    Traverse glycan tree from root and build AF3 notation string.
    """
    string = ""
    children = tree.get(node, [])
    
    if replace_root:
        base = "MAN"
    else:
        base = node.resname

    if len(children) == 0:
        return base, num + 1

    child_strings = []
    for child in children:
        s, num = traverse(tree, child, d+1, num)
        child_strings.append(s)

    child_str = "-".join(sorted(child_strings))
    return f"{base}({child_str})", num

def get_af3(g):
    print("Getting Glycan Tree...")
    af3_string, num = get_tree(g, replace_root=True)
    return af3_string


def extract_glycan_data(filename, is_strict=True, prot_name="test_prot", job_types=None):
    """
    job_types: list of str, e.g. ["w_glycans", "wo_glycans", "large_glycans"]
               or None to mean "all types"
    """
    prots = gl.Protein.from_cif(filename)
    prots.find_glycans(strict=is_strict)
    protein_chain_dict = prots.get_sequence()
    glycan_dict = prots.get_glycans()

    glycans_per_chain = {}

    for atom, glycan in glycan_dict.items():
        glycan_residue = atom.get_parent()
        protein_chain = glycan_residue.get_parent()
        chain_name = protein_chain.id

        af3 = get_af3(glycan)
        attached_index = protein_chain.get_list().index(glycan_residue) + 1

        if chain_name not in glycans_per_chain:
            glycans_per_chain[chain_name] = []

        glycans_per_chain[chain_name].append({
            "residues": af3,
            "position": attached_index
        })

    protein_chains_list_w_glycans = []
    protein_chains_list_wo_glycans = []
    protein_chains_list_large_glycans = []

    for chain, sequence in protein_chain_dict.items():
        if sequence == "":
            continue

        chain_data_with_glycans = {
            "sequence": sequence,
            "glycans": glycans_per_chain.get(chain.id, []),
            "count": 1,
        }

        chain_data_without_glycans = {
            "sequence": sequence,
            "count": 1,
        }

        max_glycans_list = copy.deepcopy(glycans_per_chain.get(chain.id, []))
        for dictionary in max_glycans_list:
            dictionary["residues"] = LONGEST_GLYCAN_RESIDUE

        chain_data_large_glycans = {
            "sequence": sequence,
            "glycans": max_glycans_list,
            "count": 1
        }

        protein_chains_list_w_glycans.append({
            "proteinChain": chain_data_with_glycans
        })

        protein_chains_list_wo_glycans.append({
            "proteinChain": chain_data_without_glycans
        })

        protein_chains_list_large_glycans.append({
            "proteinChain": chain_data_large_glycans
        })

    all_jobs = {
        "w_glycans": {
            "name": f"{prot_name}_w_glycans",
            "modelSeeds": [],
            "sequences": protein_chains_list_w_glycans
        },
        "wo_glycans": {
            "name": f"{prot_name}_wo_glycans",
            "modelSeeds": [],
            "sequences": protein_chains_list_wo_glycans
        },
        "large_glycans": {
            "name": f"{prot_name}_large_glycans",
            "modelSeeds": [],
            "sequences": protein_chains_list_large_glycans
        }
    }
   

    if job_types is None:
        # default: return all jobs
        jobs = list(all_jobs.values())
    else:
        jobs = [all_jobs[job_type] for job_type in job_types]

    return jobs


# ----------------------------------------
# NEW SCRIPT ENTRY POINT
# ----------------------------------------

def create_seeds(num_seeds):
    return [random.randint(1, 2**31 - 1) for _ in range(num_seeds)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_config", required=False, default="configs/predict/af3_webserver_run.yaml",
        help="Path to YAML file with model-specific predict parameters."
    )
    # parser.add_argument(
    #     "--cif_folder", required=True,
    #     help="Name of the CIF folder under raw_data/. E.g. mmcif_glycoprotein_subset_of_pdb"
    # )
    # parser.add_argument(
    #     "--job_types", required=False, default=None, nargs='*',
    #     help="The types of jobs to generate, e.g. 'w_glycans,wo_glycans,large_glycans'. If not provided, all types are generated."
    # )
    args = parser.parse_args()

    # Load predict config
    with open(args.predict_config) as f:
        predict_config = yaml.safe_load(f)

    num_seeds = predict_config.get("num_seeds", 1)
    dialect = predict_config.get("dialect", "alphafold3")
    version = predict_config.get("version", 1)
    job_types = predict_config.get("job_types", None)

    dataset_dir = predict_config.get("cif_folder")

    confirmed_path = PREPROCESS_DIRECTORY / predict_config.get("model_name") / "confirmed_structures.txt"
    

    cif_folder_path = RAW_DATA_DIRECTORY / dataset_dir
    output_json_dir = OUTPUT_DIRECTORY / "predict" / "af3_webserver" / dataset_dir / "jsons"
    output_json_dir.mkdir(parents=True, exist_ok=True)

    with open(confirmed_path) as f:
        confirmed_structures = [line.strip() for line in f if line.strip()]

    for pdb_filename in tqdm(confirmed_structures, desc="Generating jobs"):
        pdb_id = pdb_filename.replace(".cif", "")
        pdb_file_path = cif_folder_path / pdb_filename

        jobs = extract_glycan_data(str(pdb_file_path), prot_name=pdb_id, is_strict=False)

        for job in jobs:
            job["modelSeeds"] = create_seeds(num_seeds) if num_seeds else []
            job["dialect"] = dialect
            job["version"] = version

            job_filename = f"{job['name']}.json"
            out_path = output_json_dir / job_filename
            with open(out_path, "w") as f:
                json.dump(job, f, indent=2)

    print(f"✅ JSON jobs written to: {output_json_dir}")
