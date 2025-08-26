import os
import warnings
import json
import yaml
import re
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import cdist
import glycosylator as gl
import itertools
import string
import argparse
from tqdm import tqdm
from constants.paths import RAW_DATA_DIRECTORY, PREPROCESS_DIRECTORY

# -------------------------------
# Utility Functions
# -------------------------------

""" 
preprocess_structures.py --model_config <model_config.yaml>

"""


def are_chains_interacting(chain1, chain2, distance_threshold=5):
    coords_chain1 = chain1.get_coords()
    coords_chain2 = chain2.get_coords()
    distances = cdist(coords_chain1, coords_chain2)
    return (distances < distance_threshold).any()

def next_missing_letter(letters_set):
    alphabet = string.ascii_uppercase
    for length in range(1, len(letters_set) + 3):
        for letter_tuple in itertools.product(alphabet, repeat=length):
            letter = ''.join(letter_tuple)
            if letter not in letters_set:
                return letter

def glycan_is_supported(glycan_codes, config):
    """
    Checks:
      - only allowed residue types
      - maximum number of residues
    """

    if config.get("supported_glycans") == None: # no list of supported glycans means all are supported
        return True

    unsupported_glycans = set(glycan_codes) - set(config["supported_glycans"])
    
    raw_limit = config.get("max_glycan_residues", None)
    if raw_limit is None:
        max_limit = float("inf")
    elif isinstance(raw_limit, str) and raw_limit.lower() == "infinity":
        max_limit = float("inf")
    else:
        max_limit = raw_limit

    valid_number = len(glycan_codes) <= max_limit

    return len(unsupported_glycans) == 0 and valid_number

# -------------------------------
# Main File Processor
# -------------------------------

def process_glycoprotein_file(file_path, config, distance_threshold, file_type):
    """
    Process a single CIF file:
      - parse sequences
      - parse glycans
      - check model constraints
      - return json_data, passes_filters, filter_reasons
    """

    json_data = {
        "sequence_list": [],
        "bonded_atom_pairs": [],
        "passes_filters": False,
        "filter_reasons": []
    }

    filter_reasons = []

    try:
        if file_type == "cif":
            structure = gl.Protein.from_cif(str(file_path))
        elif file_type == "pdb":
            structure = gl.Protein.from_pdb(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        structure.find_glycans(strict=False)

        protein_chain_dict = structure.get_sequence()
        glycan_dict = structure.get_glycans()

        # -----------------------------
        # Assemble sequences
        # -----------------------------

        sequence_list = []
        bonded_atom_pairs_list = []
        ids_set = set()

        for chain, sequence in protein_chain_dict.items():
            if len(sequence) != 0:
                protein = {
                    "protein": {
                        "id": chain.id,
                        "sequence": sequence
                    }
                }
                sequence_list.append(protein)
                ids_set.add(chain.id)

        for aa_atom, glycan in glycan_dict.items():
            glycan_id = next_missing_letter(ids_set)
            ids_set.add(glycan_id)

            glycan_codes = sorted(list(res.resname for res in glycan.get_residues()))
            ligand_obj = {
                "ligand": {
                    "id": glycan_id,
                    "ccdCodes": glycan_codes
                }
            }
            sequence_list.append(ligand_obj)

            for bond in glycan.get_residue_connections():
                a1 = bond.atom1
                res1_id = a1.get_parent().serial_number
                a2 = bond.atom2
                res2_id = a2.get_parent().serial_number
                bonded_atom_pair = [[glycan_id, res1_id, a1.id],
                                    [glycan_id, res2_id, a2.id]]
                bonded_atom_pairs_list.append(bonded_atom_pair)

            aa_res = aa_atom.get_parent()
            aa_res_id = aa_res.serial_number
            aa_chain_id = aa_res.get_parent().id

            root_atom = glycan.get_root()
            root_atom_res = root_atom.get_parent()
            root_atom_res_id = root_atom_res.serial_number

            bonded_atom_pair = [[aa_chain_id, aa_res_id, aa_atom.id],
                                [glycan_id, root_atom_res_id, root_atom.id]]
            bonded_atom_pairs_list.append(bonded_atom_pair)

        json_data["sequence_list"] = sequence_list
        json_data["bonded_atom_pairs"] = bonded_atom_pairs_list

        # -----------------------------
        # Apply Filters
        # -----------------------------

        # Check total residues
        total_length = sum(len(seq["protein"]["sequence"])
                           for seq in sequence_list
                           if "protein" in seq)

        if config.get("max_residues") and total_length > config["max_residues"]:
            filter_reasons.append(f"Exceeds max residues ({total_length} > {config['max_residues']})")

        if not config.get("allowed_multimers", True):
            protein_chains = [seq for seq in sequence_list if "protein" in seq]
            if len(protein_chains) > 1:
                filter_reasons.append("Multimers not allowed.")

        unsupported_ccds = set(config.get("unsupported_ccd_codes", []))
        for seq in sequence_list:
            if "ligand" in seq:
                glycan_codes = seq["ligand"]["ccdCodes"]
                if any(code in unsupported_ccds for code in glycan_codes):
                    filter_reasons.append(f"Contains unsupported CCD codes: {glycan_codes}")
                if not glycan_is_supported(glycan_codes, config):
                    filter_reasons.append(f"Glycan not supported: {glycan_codes}")

        # Check chain-glycan interactions
        has_interactions = False
        for aa_atom, glycan in glycan_dict.items():
            for chain, sequence in protein_chain_dict.items():
                if chain.id == aa_atom.get_parent().get_parent().id:
                    continue
                if len(sequence) == 0:
                    continue
                if are_chains_interacting(chain, glycan, distance_threshold):
                    has_interactions = True
                    break
            if has_interactions:
                break

        if config.get("require_glycan_interactions", True) and not has_interactions:
            filter_reasons.append("No glycans interacting with other chains.")

        if not filter_reasons:
            json_data["passes_filters"] = True

    except Exception as e:
        filter_reasons.append(f"Exception processing file: {type(e).__name__}: {str(e)}")

    json_data["filter_reasons"] = filter_reasons
    return json_data

def process_glycan_file(file_path, config, file_type):
        """ Assuming there's only one glycan in the file, and it is a glycan file. """
        if file_type == "cif":
            glycan = gl.Glycan.from_cif(str(file_path))
        elif file_type == "pdb":
            glycan = gl.Glycan.from_pdb(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        sequence_list = []
        bonded_atom_pairs_list = []
        ids_set = set()
        
        json_data = {
        "sequence_list": [],
        "bonded_atom_pairs": [],
        "passes_filters": False,
        "filter_reasons": []
        }


        glycan_id = next_missing_letter(ids_set)
        ids_set.add(glycan_id)

        glycan_codes = sorted(list(res.resname for res in glycan.get_residues()))
        ligand_obj = {
            "ligand": {
                "id": glycan_id,
                "ccdCodes": glycan_codes
            }
        }
        sequence_list.append(ligand_obj)

        for bond in glycan.get_residue_connections(): #Includes bonds adjacent to .
            if type(bond) == tuple:
                # If bond is a tuple, it means it's a bond adjacent to a bond connecting two residues.
                # These can be removed by setting triplet = False, but I think it's better to keep them.
                #for our purposes, we will just make them regular bonds
                bond = gl.Bond(bond[0], bond[1])

            a1 = bond.atom1
            res1_id = a1.get_parent().serial_number
            a2 = bond.atom2
            res2_id = a2.get_parent().serial_number
            bonded_atom_pair = [[glycan_id, res1_id, a1.id],
                                [glycan_id, res2_id, a2.id]]
            bonded_atom_pairs_list.append(bonded_atom_pair)

        json_data["sequence_list"] = sequence_list
        json_data["bonded_atom_pairs"] = bonded_atom_pairs_list
        json_data["passes_filters"] = glycan_is_supported(glycan_codes, config)
    
        return json_data




# -------------------------------
# Main Entry Point
# -------------------------------

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent  # folder of folder of this file
    CONFIG = ROOT / "configs/preprocess/default.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", required=False, default=f"{CONFIG}",
        help="Path to model config YAML file."
    )

    args = parser.parse_args()

    # Load model config
    with open(args.model_config) as f:
        config = yaml.safe_load(f)

    distance_threshold = config.get("distance_threshold")
    file_type = config.get("file_type")  # Default to CIF if not specified
    structure_type = config.get("structure_type", "glycoprotein")  # Default to glycoprotein if not specified
    dataset_name = config.get("dataset_name")

    structure_folder_path = Path(config.get("structure_folder") )
    output_dir = PREPROCESS_DIRECTORY / config.get("model_name") / dataset_name #output directory for the specific model config
    output_dir.mkdir(parents=True, exist_ok=True)

    confirmed_structures = []


    files = [] 

    if dataset_name == "glycoshape_pdbs_validated": # only need to process one alpha and beta structure per GSID
        for gsid in sorted({f.name.split("_")[0] for f in structure_folder_path.glob("GS*.pdb")}):
            alpha_files = sorted(structure_folder_path.glob(f"{gsid}_alpha*.pdb"))
            beta_files  = sorted(structure_folder_path.glob(f"{gsid}_beta*.pdb"))
            if alpha_files:
                files.append(str(alpha_files[0].name))
            if beta_files:
                files.append(str(beta_files[0].name))
    else:
        files = sorted(os.listdir(structure_folder_path))


    for filename in tqdm(files, desc="Processing Structures"):
        if not filename.endswith(f".{file_type}"):
            continue

        name = filename.replace(f".{file_type}", "")
        file_path = structure_folder_path / filename

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if structure_type == "glycoprotein":
                json_data = process_glycoprotein_file(
                    file_path,
                    config,
                    distance_threshold,
                    file_type
                )
            elif structure_type == "glycan":
                json_data = process_glycan_file(
                    file_path,
                    config,
                    file_type
                )
            else:
                raise ValueError(f"Unsupported structure type: {structure_type}")

            # Save JSON always
            out_path = output_dir / name
            out_path.mkdir(exist_ok=True)
            with open(out_path / "sequences.json", "w") as f:
                json.dump(json_data, f, indent=2)

            if json_data["passes_filters"]:
                confirmed_structures.append(name)

    # Write confirmed list
    if config.get("model_name") == "manifest": # no need to write confirmed structures for the default model
        print("No confirmed structures for default model, simply generate the normal manifest.")
    else:
        confirmed_file = output_dir / f"confirmed_structures_{config.get("dataset_name")}.txt"
        with open(confirmed_file, "w") as f:
            f.writelines([x + "\n" for x in confirmed_structures])
        print(f"\n✅ Wrote confirmed list: {confirmed_file}")
        print(f"✅ Total confirmed: {len(confirmed_structures)}")



