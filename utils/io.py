# utils/io.py

import shutil
import zipfile
from pathlib import Path
from constants.paths import (
    RAW_DATA_DIRECTORY,
    GLYCOSHAPE_DATA_PATH,
    ALPHAFOLD_MODELING_GLYCOSHAPE_DATA_PATH,
)


def ensure_dir(path: Path) -> None:
    """
    Ensure the directory at `path` exists; create it if necessary.
    """
    path.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a ZIP archive to `extract_to` if it's not already populated.
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    if extract_to.exists() and any(extract_to.iterdir()):  # skip if non-empty
        return
    ensure_dir(extract_to)
    shutil.unpack_archive(str(zip_path), str(extract_to))


def extract_all_zips(source_dir: Path, target_dir: Path) -> None:
    """
    Recursively extract all ZIP files under `source_dir` into `target_dir`.

    Each ZIP is unpacked into a subfolder of `target_dir` mirroring its
    relative path without the .zip suffix.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    ensure_dir(target_dir)
    print("Skipping non-empty folders while extracting files...")
    for zip_file in source_dir.rglob("*.zip"):
        rel_path = zip_file.relative_to(source_dir).with_suffix("")
        dest = target_dir / rel_path
        extract_zip(zip_file, dest)


def extract_glycoshape_files() -> None:
    """
    Unpack the main GlycoShape ZIP and all nested ZIPs into the GlycoShape data path.
    """
    base = Path(RAW_DATA_DIRECTORY) / "glycoshape"
    main_zip = base / "GlycoShape.zip"
    temp = base / "gs_extracted_from_website"
    extract_zip(main_zip, temp)
    extract_all_zips(temp, Path(GLYCOSHAPE_DATA_PATH))


def extract_alphafold_files() -> None:
    """
    Unpack all Alphafold model ZIPs into the Alphafold modeling data path.
    """
    source = Path(RAW_DATA_DIRECTORY) / "af_models" / "modeling_glycoshape" / "af_extracted_from_website"
    extract_all_zips(source, Path(ALPHAFOLD_MODELING_GLYCOSHAPE_DATA_PATH))