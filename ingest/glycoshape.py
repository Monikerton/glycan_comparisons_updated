from pathlib import Path
import os
from collections import defaultdict
import json
import pandas as pd
import shutil
import glycosylator as gl
from pymol import cmd


from ingest.common import (
    glycoshape_ROH_corrector,
    get_components,
    get_tree,
    get_angles,
    get_ref_residues,
    get_error_message,
    get_pymol_residue_ids,
    convert_glycam_to_pdb,
    create_structure_replace_glycam_code_with_pdb_code
)

from constants.paths import (
    GLYCOSHAPE_ZIP,
    GLYCOSHAPE_DATA_PATH,
    PREPROCESS_OUTPUT_DIR,
    RAW_DATA_DIRECTORY,
    CONSTANTS_DIRECTORY,
    OUTPUT_DIRECTORY
)


from utils.io import extract_glycoshape_files, ensure_dir

class GlycoShapeParser:
    def __init__(
        self,
        database_path: str,
        glycam_to_pdb_path: str,
        glycosylator_pdb_path: str,
        glycosylator_objects_path: str,
    ) -> None:
        """Initialize parser for the raw GlycoShape dataset."""
        # disable internal coordinate inference in glycosylator
        gl.utils.dont_use_ic()

        self.database_path = Path(database_path)
        self.glycam_to_pdb_path = Path(glycam_to_pdb_path)
        self.glycosylator_pdb_path = Path(glycosylator_pdb_path)
        self.glycosylator_objects_path = Path(glycosylator_objects_path)
    
    def organize_and_convert(self) -> None:
        """
        Reorganize GlycoShape folders by GSID and convert Glycam codes to PDB codes.
        Place all intermediate files under the project's output directory.
        """
        # Input JSON folders
        base_input = Path(GLYCOSHAPE_DATA_PATH)
        # Organize under output intermediates
        glycam_sub = OUTPUT_DIRECTORY / "ingest" / 'glycoshape' / 'gs_data_organized_by_gsid'
        # Final organized outputs
        output_folder = OUTPUT_DIRECTORY / "ingest" / 'glycoshape' / 'organized_with_residue_pdb_ids'

        # Step A: Reorganize folders
        ensure_dir(glycam_sub)
        for iupac_dir in base_input.iterdir():
            if not iupac_dir.is_dir():
                continue
            # Load JSON to get GSID
            json_files = list(iupac_dir.glob('*.json'))
            if len(json_files) > 1: #no json file will raise file not found error later.
                raise ValueError(f"Expected exactly one JSON file in {iupac_dir}, but found {len(json_files)}")
            with json_files[0].open("r", encoding="utf-8") as jf:
                glycan_info = json.load(jf)            
            gs_id = glycan_info.get('ID')
            if not gs_id:
                continue
            target_dir = glycam_sub / gs_id
            ensure_dir(target_dir)

            # Copy PDBs avoiding _conn
            hetatm_dir = iupac_dir / 'GLYCAM_format_HETATM'
            if not hetatm_dir.exists():
                continue
            for pdb_file in hetatm_dir.glob('*.pdb'):
                if '_conn' in pdb_file.name:
                    continue
                shutil.copy(pdb_file, target_dir / pdb_file.name)

        # Step B: Gather residue mappings via PyMOL
        residue_list = []
        for gs_dir in glycam_sub.iterdir():
            if not gs_dir.is_dir():
                continue
            for pdb_file in gs_dir.glob('*.pdb'):
                cmd.load(str(pdb_file), 'glycam_structure')
                residue_list += get_pymol_residue_ids('glycam_structure', gs_dir.name)
                cmd.delete('glycam_structure')
        glycam_codes = set((code, gid) for _, code, gid in residue_list)

        # Load conversion chart
        chart_path = Path(CONSTANTS_DIRECTORY) / 'glycam_pdb_chart.csv'
        chart = pd.read_csv(chart_path, sep='\t', index_col='One-letter Code')
        invalid = set()

        # Convert codes and identify invalid IDs
        for code, gid in glycam_codes:
            res = convert_glycam_to_pdb(code, chart)
            if res is None and code != 'ROH':
                invalid.add(gid)

        # Step C: Build final organized folder with valid IDs
        ensure_dir(output_folder)
        for gs_dir in glycam_sub.iterdir():
            if gs_dir.name in invalid:
                continue
            target = output_folder / gs_dir.name
            ensure_dir(target)
            for pdb_file in gs_dir.glob('*.pdb'):
                dest = target / (pdb_file.stem + '_conv.pdb')
                create_structure_replace_glycam_code_with_pdb_code(pdb_file, dest, chart)

    def parse(self) -> None:
        """
        1) Ensure output directories exist
        2) Extract raw archives if needed
        3) Loop through glycan JSONs to determine GS IDs
        4) Correct PDBs, run glycosylator functions, and save summaries
        5) Aggregate into a single pickle
        """
        # 1) Prepare output structure
        ensure_dir(self.glycam_to_pdb_path)
        ensure_dir(self.glycosylator_pdb_path)
        ensure_dir(self.glycosylator_objects_path.parent)

        # 2) Unpack raw GlycoShape zips
        print("Extracting Glycoshape Files...")
        extract_glycoshape_files()

        print("Reorganizing Files")
        self.organize_and_convert()

        # 3) Initialize counters and storage
        failures = defaultdict(int)
        success = 0
        glycan_objects = []
        ref_residues = get_ref_residues()

        # 4) Iterate through folders in database_path
        for folder in sorted(self.database_path.iterdir()):
            if not folder.is_dir():
                continue
            json_file = folder / f"{folder.name}.json"
            try:
                with open(json_file) as f:
                    gly_dict = json.load(f)
            except FileNotFoundError:
                failures['FileNotFoundError'] += 1
                continue

            gs_id = gly_dict.get('ID')
            
            if not gs_id:
                failures['MissingID'] += 1
                continue

            # Locate converted PDBs for this GS ID
            sub_folder = self.glycam_to_pdb_path / gs_id
            if not sub_folder.exists():
                failures['NoConvertedPDBs'] += 1
                continue

            # Create per-GS output directories
            out_raw = self.glycam_to_pdb_path / gs_id
            out_corr = self.glycosylator_pdb_path / gs_id
            ensure_dir(out_raw)
            ensure_dir(out_corr)

            # 5) Process each PDB file
            for pdb_file in sorted(sub_folder.glob("*.pdb")):
                try:
                    print("PDB FILE", pdb_file)
                    splits = pdb_file.stem.split("_")
                    cluster = splits[-3]
                    configuration = splits[-2]
                    key = f"{cluster}_{configuration}"

                    # Correct ROH and write to raw output
                    corrected_pdb = out_raw / f"{key}.pdb"
                    glycoshape_ROH_corrector(pdb_file, corrected_pdb)
                    print("PDB HERE", corrected_pdb)
                    g = gl.Glycan.from_pdb(str(corrected_pdb)) #put the glycan in the glycoshape format just to make sure everything is converted correctly
                    
                    g.infer_bonds(max_bond_length=1.622, restrict_residues=True)
                    g.infer_residue_connections(bond_length=1.62)
                    g.infer_glycan_tree()
                    computed_iupac = g.to_iupac()
                    
                    g.to_pdb(out_corr / f"{key}.pdb")
                    

                    # Compute components, tree, angles
                    components = get_components(corrected_pdb)
                    af3_string, num = get_tree(g)
                    
                    # angles_dict = get_angles(g, ref_residues)
                    
                    # ## Added code for calcuating dihedral angles for each of the clusters
                    # print("\n Adding code for Dihedral Angle Calculation...")
                                        
                    # for residue_connection, angles_values in angles_dict.items():
                    #     angles_list = angles_values["phi"] + angles_values["psi"] + angles_values["omega"]
                    #     dataframe_dict = {"Glycoshape ID": gs_id, "Computed IUPAC": computed_iupac, "Cluster": cluster, "Configuration": configuration,  "Glycan Connection": residue_connection, "Angles": angles_values, "Angles List": angles_list}
                    #     dihedral_angles_output.append(dataframe_dict)
                

                    # Build summary dict
                    new_dict = {
                        'configuration': configuration,
                        'components': components,
                        'tree': af3_string,
                        'iupac': computed_iupac,
                        'detailed': {},
                    }

                    # Write JSON summary
                    summary_file = out_corr / f"{cluster}_{configuration}_summary.json"
                    with open(summary_file, 'w') as jf:
                        json.dump(new_dict, jf, indent=2)

                    # Collect for aggregation
                    glycan_objects.append((gs_id, configuration, cluster, g))
                    success += 1

                except RecursionError as e:
                    failures['RecursionError'] += 1
                    get_error_message(e)
                except Exception as e:
                    failures['Other'] += 1
                    get_error_message(e)

        # 6) Aggregate pickled DataFrame
        df = pd.DataFrame(
            glycan_objects,
            columns=['GSID', 'Configuration', 'Cluster', 'Glycan Obj'],
        )
        df.to_pickle(self.glycosylator_objects_path)

        print(f"GlycoShape parsing complete: {success} successes, failures: {dict(failures)}")

def run(params: dict) -> None:
    """
    Standard interface for this module.

    Parameters
    ----------
    input_path : Path
        Raw GlycoShape database directory.
    output_path : Path
        Base output directory for corrected PDBs.
    params : dict
        Must contain:
          - glycam_to_pdb_path
          - glycosylator_pdb_path
          - glycosylator_objects_path
    """
    
    print("Parsing Glycoshape files...")
    parser = GlycoShapeParser(
        database_path=params['source_dir'],
        glycam_to_pdb_path=params["glycam_to_pdb_path"],
        glycosylator_pdb_path=params["glycosylator_pdb_path"],
        glycosylator_objects_path=params["glycosylator_objects_path"],
    )
    parser.parse()