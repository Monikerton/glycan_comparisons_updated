""" This module contains functions to convert coordinate structures to DataFrame format, stored in pkl files.
Useful for processing and analyzing molecular structures in different datasets, ensuring standardized formats.
 """

import pandas as pd
from pathlib import Path
import glycosylator as gl
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore", ResourceWarning)


aa_residues = {"GLY", "SER", "ASN"}
atom_mapping = {"C2N": "C7", "O2N": "O7", "CME": "C8"} #instead of using the autolabel function, used these because they were a lot faster



# folder_path = Path("output/processed/alphafold_webserver_glycoshape_cifs_validated")  # Path to the folder containing structure files
# dataset_name = "alphafold_webserver_glycoshape_cifs_validated"  # Change this to the dataset name you are working with
# file_type = "cif"  # Change to "cif" or "pdb" as needed
# structure_paths = list(folder_path.glob(f"*.{file_type}"))  # Get all structure files in the folder
# output_path = Path("output/metrics/glycans/alphafold_webserver_glycoshape_coordinates.pqt")  # Path to save the DataFrame
# output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
# offset = 0  # Offset for residue numbering (default is 0)



folder_path = Path("output/processed/glycoshape_pdbs_validated")  # Path to the folder containing structure files
dataset_name = "glycoshape_pdbs_validated"  # Change this to the dataset name you are working with
file_type = "pdb"  # Change to "cif" or "pdb" as needed
structure_paths = list(folder_path.glob(f"*.{file_type}"))  # Get all structure files in the folder
output_path = Path("output/metrics/glycans/glycoshape_validated_coordinates.pqt")  # Path to save the DataFrame
output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
offset = -1 #glycoshape starts residue numbering at 2 for some reason.







#extract the coordiates of a glycan via a dataframe

def extract_glycan_coordinates(gs_id, configuration, cluster, glycan, offset=0):
    """Extracts coordinates of atoms in a glycan and returns them as a DataFrame.
    gs_id: GlycoShape ID
    configuration: Configuration of the glycan (alpha or beta)
    cluster: Cluster number of the glycan
    glycan: The Glycosylator object containing residues and atoms
    offset: Offset for residue numbering (default is 0)
    """
    
    atom_data = []
    for residue in glycan.residues:  # Iterate over residues in the glycan
        if residue.resname in aa_residues: continue # don't need to worry about amino acids (needed as output for some prediction models)
        for atom in residue.get_atoms():  # Iterate over atoms in the residue
            if atom.element == "H": continue # We don't care about coordinates of hydrogen atoms
            atom_name = atom.name
            if atom_name in atom_mapping:
                atom_name = atom_mapping[atom_name]

            coords = atom.get_coord()  # Assuming atom.coords gives a numpy array [X, Y, Z]
            atom_data.append({
                "GSID": gs_id,
                "Configuration": configuration,
                "Cluster": cluster,
                "Resname": residue.resname,  # Assuming residue has a 'name' attribute
                "Resid": residue.serial_number + offset,  # Residue ID is the serial number of the residue
                #Residue ID is for the offset of the names of the residues
                "Atom Name": atom_name,
                "X": coords[0],
                "Y": coords[1],
                "Z": coords[2]
            })
    
    return pd.DataFrame(atom_data)

def save_glycan_coordinates_df(df, output_path):
    """Saves the DataFrame of glycan coordinates to a pickle file.
    df: DataFrame containing glycan coordinates
    output_path: Path where the DataFrame will be saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    df.to_parquet(str(output_path), index=False)  # Save DataFrame as a pickle file
    print(f"Glycan coordinates saved to {output_path}")


df_list = []

for structure_path in tqdm(structure_paths):
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

    glycan_df = extract_glycan_coordinates(gs_id, configuration, cluster, g, offset=offset)
    df_list.append(glycan_df)

final_df = pd.concat(df_list, ignore_index=True)

save_glycan_coordinates_df(final_df, output_path)  # Save the DataFrame to a parquet file