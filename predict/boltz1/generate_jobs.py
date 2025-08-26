import json
import glycosylator as gl
from constants.paths import PREPROCESS_DIRECTORY, OUTPUT_DIRECTORY, RAW_DATA_DIRECTORY
import os
import string
import itertools
import random
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm


"""
generate_jobs.py --predict_config <predict_config.yaml> --cif_folder <cif_folder_name> (under preprocess/ and raw_data)

"""

# Conversion function
def convert_to_yaml_format(data):
    yaml_data = {
        'sequences': [],
        'constraints': []
    }

    # Convert sequences
    for entity in data['sequence_list']:
        if 'protein' in entity: # no corresponding sets for dna or rna
            protein_data = entity['protein']
            yaml_data['sequences'].append({
                'protein': {
                    'id': protein_data['id'],
                    'sequence': protein_data['sequence'],
                    'modifications': []  # Placeholder for modifications if needed
                }
            })
        elif 'ligand' in entity:
            ligand_data = entity['ligand']
            for i, ccd in enumerate(ligand_data['ccdCodes']): #enumerate to handle multiple ligands in separate entries
                # Assuming ligand_data['id'] is a list of IDs for each ligand
                yaml_data['sequences'].append({
                    'ligand': {
                        'id': f"{ligand_data['id']}_{i}",
                        'ccd': ccd, 
                        'modifications': []  # Placeholder for modifications if needed
                    }
                })
        else:
            raise ValueError("Unknown entity type in sequence_list")

    # Convert bondedAtomPairs to constraints
    for bond in data['bonded_atom_pairs']:
        yaml_data['constraints'].append({
            'bond': {
                'atom1': bond[0],
                'atom2': bond[1]
            }
        })

    return yaml_data


# Convert and print as YAML

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_config", required=False, default="configs/predict/boltz1.yaml",
        help="Path to YAML file with model-specific predict parameters."
    )

    args = parser.parse_args()

    # Load predict config
    with open(args.predict_config) as f:
        predict_config = yaml.safe_load(f)

    dataset_dir = predict_config.get("cif_folder")

    # confirmed_path = PREPROCESS_DIRECTORY / predict_config.get("model_name") / "confirmed_structures.txt"

    # with open(confirmed_path) as f:
    #     confirmed_structures = [line.strip() for line in f if line.strip()]

    manifest_path = PREPROCESS_DIRECTORY / "manifest" / dataset_dir

    confirmed_structures = [p.name for p in manifest_path.iterdir() if p.is_dir()] # get the id of the confirmed structures

    output_yaml_dir = OUTPUT_DIRECTORY / "predict" / "boltz1" / dataset_dir / "yamls"
    output_yaml_dir.mkdir(parents=True, exist_ok=True)



    for pdb_filename in tqdm(confirmed_structures, desc="Generating jobs"):
        pdb_id = pdb_filename.replace(".cif", "")
        pdb_file_path = manifest_path / pdb_filename / "sequences.json"

        # Load sequence JSON
        with open(pdb_file_path) as f:
            data = json.load(f)

        # Convert to YAML format
        yaml_data = convert_to_yaml_format(data)


        out_path = output_yaml_dir / f"{pdb_id.replace('.cif','')}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)


    print(f"✅ YAML jobs written to: {output_yaml_dir}")
