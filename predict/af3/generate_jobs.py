import os
import json
import yaml
import random
from pathlib import Path
import argparse
from tqdm import tqdm
from constants.paths import PREPROCESS_DIRECTORY, OUTPUT_DIRECTORY

"""

generate_jobs.py --predict_config <predict_config.yaml> --cif_folder <cif_folder_name> (under preprocess/)

This script generates JSON job files for AlphaFold3 predictions based on sequences and model seeds."""

def create_alphafold_json(name, model_seeds, sequences,
                          bonded_atom_pairs=None,
                          dialect='alphafold3', version=1):
    return {
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": sequences,
        "bondedAtomPairs": bonded_atom_pairs,
        "dialect": dialect,
        "version": version
    }

def create_seeds(num_seeds):
    return [random.randint(1, 2**31 - 1) for _ in range(num_seeds)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_config", required=False, default="configs/predict/af3.yaml",
        help="Path to YAML file with model-specific predict parameters."
    )

    args = parser.parse_args()

    # Load predict config
    with open(args.predict_config) as f:
        predict_config = yaml.safe_load(f)

    num_seeds = predict_config.get("num_seeds", None)
    dialect = predict_config.get("dialect", "alphafold3")
    version = predict_config.get("version", 3)
    dataset_dir = predict_config.get("cif_folder", "default_dataset")

    # Paths
    output_predict_dir = OUTPUT_DIRECTORY / "predict/af3" / dataset_dir / "jsons"


    # confirmed_path = PREPROCESS_DIRECTORY / dataset_dir / "confirmed_structures.txt"
    # # Read confirmed list
    # with open(confirmed_path) as f:
    #     confirmed_structures = [line.strip() for line in f if line.strip()]



    preprocess_base = PREPROCESS_DIRECTORY / "manifest" / dataset_dir
    output_json_dir = output_predict_dir / dataset_dir / "jsons"
    output_json_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = PREPROCESS_DIRECTORY / "manifest" / dataset_dir

    confirmed_structures = [p.name for p in manifest_path.iterdir() if p.is_dir()] # get the id of the confirmed structures


    for pdb_id in tqdm(confirmed_structures, desc="Generating job JSONs"):
        json_input_path = preprocess_base / pdb_id.replace(".cif", "") / "sequences.json"
        
        # Load sequence JSON
        with open(json_input_path) as f:
            data = json.load(f)

        # Only generate jobs for structures that passed filters
        if not data.get("passes_filters", False):
            continue

        sequence_list = data["sequence_list"]
        bonded_atom_pairs = data["bonded_atom_pairs"]

        job_json = create_alphafold_json(
            name=f"{pdb_id}_alphafold_jobs",
            model_seeds=create_seeds(num_seeds) if num_seeds else [],
            sequences=sequence_list,
            bonded_atom_pairs=bonded_atom_pairs,
            dialect=dialect,
            version=version
        )

        out_path = output_json_dir / f"{pdb_id.replace('.cif','')}.json"
        with open(out_path, "w") as f:
            json.dump(job_json, f, indent=2)

    print(f"✅ Written {len(confirmed_structures)} JSON job files to {output_json_dir}")
