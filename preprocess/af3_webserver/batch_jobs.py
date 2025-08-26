import json
import os
from pathlib import Path

BATCH_SIZE = 100

def batch_jobs(
    id_txt_path,
    json_folder_path,
    output_folder_path,
):
    # Load list of IDs from the text file
    with open(id_txt_path) as f:
        ids = [line.strip() for line in f if line.strip()]

    json_folder = Path(json_folder_path)
    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_jobs = []

    for pdb_id in ids:
        # Find all JSON files that start with this pdb_id
        json_files = list(json_folder.glob(f"{pdb_id}*.json"))

        if not json_files:
            print(f"WARNING: No JSON files found for ID: {pdb_id}")
            continue

        for json_file in json_files:
            with open(json_file) as jf:
                data = json.load(jf)

                # Each JSON might be:
                # - a list of jobs
                # - a single job dict
                if isinstance(data, list):
                    all_jobs.extend(data)
                elif isinstance(data, dict):
                    all_jobs.append(data)
                else:
                    print(f"ERROR: Unexpected JSON format in {json_file}")
                    continue

    # Split into batches
    for i in range(0, len(all_jobs), BATCH_SIZE):
        batch = all_jobs[i : i + BATCH_SIZE]

        batch_filename = f"batch_{i // BATCH_SIZE:03}.json"
        batch_path = output_folder / batch_filename

        with open(batch_path, "w") as f:
            json.dump(batch, f, indent=2)

        print(f"Wrote batch: {batch_path}")
