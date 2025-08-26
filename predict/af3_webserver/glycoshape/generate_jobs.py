import os
import json
import yaml
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def create_seeds(num_seeds):
    return [random.randint(1, 2**31 - 1) for _ in range(num_seeds)]

""""
A special version of generate_jobs, where one 
generates Alphafold Webserver jobs for glycans from Glycoshape specifically (has the alpha / beta confirmations)

WIP
"""

def generate_glycoshape_jobs(alpha_string, beta_string, gs_id, num_seeds, dialect, version):
    """
    Build two jobs: alpha and beta conformation.
    """
    jobs = []

    for conf, af3_string in [("alpha", alpha_string), ("beta", beta_string)]:
        job = {
            "name": f"{gs_id}_{conf}",
            "modelSeeds": create_seeds(num_seeds),
            "sequences": [
                {
                    "ligand": {
                        "id": "LIG1",
                        "ccdCodes": parse_af3_string(af3_string)
                    }
                }
            ],
            "bondedAtomPairs": [],
            "dialect": dialect,
            "version": version
        }
        jobs.append(job)

    return jobs

def parse_af3_string(af3_string):
    """
    Parse AF3 notation into a list of CCD codes.
    E.g. MAN(MAN(MAN)) → ['MAN', 'MAN', 'MAN']
    """
    if not af3_string:
        return []
    
    # Extract all residue codes (uppercase letters, numbers allowed)
    import re
    matches = re.findall(r'[A-Z0-9]+', af3_string)
    return matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_config", required=True,
        help="Path to YAML file with predict parameters."
    )
    parser.add_argument(
        "--glycoshape_folder", required=True,
        help="Subfolder under raw_data/ containing GlycoShape results."
    )
    args = parser.parse_args()

    # Load config
    with open(args.predict_config) as f:
        predict_config = yaml.safe_load(f)

    num_seeds = predict_config.get("num_seeds", 1)
    dialect = predict_config.get("dialect", "glycoshape")
    version = predict_config.get("version", 1)

    RAW_DATA_DIRECTORY = Path("raw_data")
    OUTPUT_DIRECTORY = Path("output/predict/glycoshape")

    glycoshape_base = RAW_DATA_DIRECTORY / args.glycoshape_folder
    output_json_dir = OUTPUT_DIRECTORY / args.glycoshape_folder / "jsons"
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------
    # Integrated glycan dictionary building code
