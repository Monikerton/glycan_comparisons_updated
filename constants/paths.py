
from pathlib import Path

# Base directories (relative to project root)
RAW_DATA_DIRECTORY = Path("raw_data")
OUTPUT_DIRECTORY = Path("output")
CONSTANTS_DIRECTORY = Path("constants")
PREDICT_DIRECTORY =  Path("predict")
PREPROCESS_DIRECTORY = Path("preprocess")


PREPROCESS_OUTPUT_DIR = OUTPUT_DIRECTORY / "preprocess"

# GlycoShape raw paths
GLYCOSHAPE_ZIP = RAW_DATA_DIRECTORY / "glycoshape" / "GlycoShape.zip"
GLYCOSHAPE_EXTRACT_TMP = RAW_DATA_DIRECTORY / "glycoshape" / "gs_extracted_from_website"
GLYCOSHAPE_DATA_PATH = RAW_DATA_DIRECTORY / "glycoshape" / "gs_extracted_from_website_unzipped"

# Alphafold modeling GlycoShape raw paths
ALPHAFOLD_ZIP_DIR = RAW_DATA_DIRECTORY / "af_models" / "modeling_glycoshape" / "af_extracted_from_website"
ALPHAFOLD_MODELING_GLYCOSHAPE_DATA_PATH = RAW_DATA_DIRECTORY / "af_models" / "modeling_glycoshape" / "af_extracted_from_website_unzipped"

# Preprocess output directories
# Corrected PDBs and summaries
GLYCOSHAPE_PDB_OUTPUT = OUTPUT_DIRECTORY / "ingest" / "glycoshape" / "glycoshape_pdbs"

# Paths for aggregated GlycoShape objects
GLYCOSHAPE_OBJECTS_PICKLE = OUTPUT_DIRECTORY / "ingest" / "glycoshape" / "glycoshape_objects.pkl"
