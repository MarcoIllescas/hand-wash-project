"""
Management of the processed videos registry.
Maintains a CSV with the relation:
    processed_id ↔ original_filename ↔ processing metadata
"""
import os
import csv
from datetime import datetime

REGISTRY_FILENAME = "registry.csv"

FIELDNAMES = [
    "processed_id",
    "original_filename",
    "processed_date",
    "fps_original",
    "fps_new",
    "resolution_original",
    "resolution_new",
    "frames_original",
    "frames_saved",
    "signature_saved",
]

# ------------------------------------------------------------------- #
#                       CSV reading and writing                       #
# ------------------------------------------------------------------- #
def _registry_path(data_dir: str) -> str:
    return os.path.join(data_dir, REGISTRY_FILENAME)


def load_registry(data_dir: str) -> list[dict]:
    """
    Loads the existing registry. Returns an empty list if it does not exist.
    """
    path = _registry_path(data_dir)
    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_registry(data_dir: str, records: list[dict]) -> None:
    """
    Overwrites the CSV with the current list of records.
    """
    path = _registry_path(data_dir)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

# ------------------------------------------------------------------- #
#                      Generation of the next ID                      #
# ------------------------------------------------------------------- #
def next_id(data_dir: str) -> str:
    """
    Reads the existing IDs in the registry and returns the next one
    in format hw_NNNNN (5 digits with leading zeros).

    Examples:
        No previous records → "hw_00001"
        Last ID "hw_00042" → "hw_00043"
    """
    records = load_registry(data_dir)

    if not records:
        return "hw_00001"

    numbers = []
    for r in records:
        try:
            num = int(r["processed_id"].split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    next_number = max(numbers) + 1 if numbers else 1
    
    return f"hw_{next_number:05d}"

# ------------------------------------------------------------------- #
#                   Add a new video to the registry                   #
# ------------------------------------------------------------------- #
def register_video(
    data_dir: str,
    original_filename: str,
    preprocessing_metadata: dict,
    notes: str = ""
) -> str:
    """
    Adds a new video to the registry and returns its processed_id.

    Parameters:
        data_dir              : root data directory (where registry.csv lives)
        original_filename     : original video filename (e.g., "juan_perez.mp4")
        preprocessing_metadata: dict returned by preprocess_video()
        notes                 : optional free text

    Returns:
        assigned processed_id (e.g., "hw_00003")
    """
    records   = load_registry(data_dir)
    new_id    = next_id(data_dir)

    new_record = {
        "processed_id"       : new_id,
        "original_filename"  : original_filename,
        "processed_date"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fps_original"       : preprocessing_metadata.get("fps_original", ""),
        "fps_new"            : preprocessing_metadata.get("fps_new", ""),
        "resolution_original": str(preprocessing_metadata.get("resolution_original", "")),
        "resolution_new"     : str(preprocessing_metadata.get("resolution_new", "")),
        "frames_original"    : preprocessing_metadata.get("frames_original", ""),
        "frames_saved"       : preprocessing_metadata.get("frames_saved", ""),
        "signature_saved"    : "False",
    }

    records.append(new_record)
    save_registry(data_dir, records)

    return new_id

def mark_signature_saved(data_dir: str, processed_id: str) -> None:
    """
    Updates the 'signature_saved' field to True for the indicated video.
    Called after signature_builder successfully saves the .npy file.
    """
    records = load_registry(data_dir)

    for r in records:
        if r["processed_id"] == processed_id:
            r["signature_saved"] = "True"
            break

    save_registry(data_dir, records)