"""
This codes calculates dihedral angles for glycan structures in a folder, and stores them in a DataFrame.

"""
from ingest.common import get_angles, get_ref_residues
import glycosylator as gl
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore", ResourceWarning)


ref_residues = get_ref_residues()

folder_path = Path("output/processed/alphafold_webserver_glycoshape_cifs_validated")
dataset_name = "alphafold_webserver_glycoshape_cifs_validated"  # Change this to the dataset name you are working with
file_type = "cif"  # Change to "cif" or "pdb" as needed
structure_paths = list(folder_path.glob(f"*.{file_type}"))  # Get all structure files in the folder
output_path = Path("output/metrics/glycans/alphafold_webserver_glycoshape_dihedral_angles.pqt")


dihedral_angles_output = []  # List to store DataFrame dictionaries


for structure_path in tqdm(structure_paths, desc="Calculating Dihedral Angles"):
    if file_type == "cif":
        g = gl.Glycan.from_cif(str(structure_path))
    elif file_type == "pdb":
        g = gl.Glycan.from_pdb(str(structure_path))
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    if dataset_name == "glycoshape_pdbs_validated" or dataset_name == "alphafold_webserver_glycoshape_cifs_validated": #maybe make this a pass
        gs_id = structure_path.stem.split("_")[0]  # Extract GSID from the filename
        configuration = structure_path.stem.split("_")[1]  # Extract configuration (alpha/beta)
        cluster = int(structure_path.stem.split("_")[2].removeprefix("cluster"))  # Extract cluster number

    if g is not None:
        g.infer_bonds(max_bond_length=1.622, restrict_residues=True)
        g.infer_residue_connections(bond_length=1.62)  # Use the correct bond length
        g.infer_glycan_tree()

    try:
        angles_dict = get_angles(g, ref_residues)
    except Exception as e:
        print(f"Error calculating angles for {structure_path}: {e}")
        continue
                    
## Added code for calcuating dihedral angles for each of the clusters
    # print("\n Adding code for Dihedral Angle Calculation...")
                        
    for residue_connection, angles_values in angles_dict.items():
        angles_list = angles_values["phi"] + angles_values["psi"] + angles_values["omega"]
        dataframe_dict = {"Glycoshape ID": gs_id, "Cluster": cluster, "Configuration": configuration,  "Residue Connection": residue_connection, "Angles List": angles_list}
        dihedral_angles_output.append(dataframe_dict)

# Convert the list of dictionaries to a DataFrame, return it
dihedral_angles_df = pd.DataFrame(dihedral_angles_output)
dihedral_angles_df.to_parquet(output_path, index=False)  # Save DataFrame as a pickle file