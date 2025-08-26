import glycosylator as gl
import string
import itertools

glycan_file = "/data/rbg/users/dkwabiad/temp_while_cp_rsg/glycan_comp_restructure/raw_data/filtered_glycoshape_pdbs/GS00002_alpha.pdb"

glycan = gl.Glycan.from_pdb(glycan_file)
print(glycan.get_residue_connections())