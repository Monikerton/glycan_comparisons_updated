""" This code flattens the AlphaFold CIF files by extracting the GSID, configuration, and cluster information from the filenames.
And puts the reulting files in the "processed" directory.
"""

from pathlib import Path
import shutil

src = Path("output/processed/alphafold_webserver_output/modeling_glycoshape/af_extracted_from_website_unzipped")
for sub_dir in src.iterdir():
    if not sub_dir.is_dir():
        continue
    for file in sub_dir.glob("*.cif"):
        # Extract GSID, configuration, and cluster from the filename
        parts = file.stem.split("_")
        if len(parts) < 3:
            print(f"Skipping {file} due to unexpected filename format.")
            continue
        
        gs_id = parts[1].upper()
        configuration = parts[2]
        cluster = parts[6]
        
        # Create a new filename with the extracted information
        new_filename = f"{gs_id}_{configuration}_cluster{cluster}.cif"
        
        # Define the destination path
        dst_path = Path("output/processed/alphafold_webserver_cifs_validated") / new_filename
        
        # Ensure the destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file to the new location with the new name
        shutil.copy(file, dst_path)
        print(f"Copied {file} to {dst_path}")
