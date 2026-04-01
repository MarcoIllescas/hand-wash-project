"""
Full workflow per video:
    1. Preprocess video (FPS, resolution, CLAHE)
    2. Register in registry.csv with ID hw_NNNNN
    3. Extract distinctive frames (SSIM)
    4. Detect landmarks with MediaPipe
    5. Interpolate missing landmarks
    6. Compute geometric features
    7. Build temporal signature (normalize + smooth)
    8. Save signature .npy  
    9. Mark signature as saved in registry.csv
"""
import os
import numpy as np
from pipeline.preprocessor import preprocess_video
from pipeline.extractor import create_detector, extract_video_landmarks
from pipeline.builder import build_signature, normalize_signature, smooth_signature
from utils.registry_manager import register_video, mark_signature_saved, load_registry

# ------------------------------------------------------------------- #
#                        General configuration                        #
# ------------------------------------------------------------------- #
CONFIG = {
    # Paths
    "raw_dir"        : "data/raw",
    "processed_dir"  : "data/processed",
    "signatures_dir" : "data/signatures",
    "data_dir"       : "data",

    # Preprocessing
    "target_fps"     : 25,
    "target_size"    : (640, 480),
    "apply_clahe"    : True,

    # Landmark extraction
    "model_path"     : "hand_landmarker.task",
    "num_hands"      : 2,
    "ssim_threshold" : 0.88,

    # Signature
    "smoothing_sigma": 1.0,
}

# ------------------------------------------------------------------- #
#                       Pipeline Test (1 video)                       #
# ------------------------------------------------------------------- #
def process_single_video(
    original_path: str,
    detector,
    config: dict = CONFIG,
    notes: str = ""
) -> dict:
    """
    Executes the full pipeline for a single video.

    Parameters:
        original_path : full path to the original video (in raw/)
        detector      : initialized HandLandmarker
        config        : configuration dictionary
        notes         : free text saved in the registry

    Returns:
        dict with:
            "processed_id"   : assigned ID (e.g., "hw_00003")
            "signature_shape": shape of the signature (T, D)
            "status"         : "ok" or "error"
            "message"        : description of the result or error
    """
    original_filename = os.path.basename(original_path)

    print(f"\n{'='*60}")
    print(f"  Processing: {original_filename}")
    print(f"{'='*60}")

    # --- STEP 1: Preprocess video --- #
    print("  [1/5] Preprocessing video...")

    temp_processed_path = os.path.join(
        config["processed_dir"],
        f"_temp_{original_filename}"
    )

    try:
        metadata = preprocess_video(
            input_path   = original_path,
            output_path  = temp_processed_path,
            target_fps   = config["target_fps"],
            target_size  = config["target_size"],
            apply_clahe  = config["apply_clahe"],
        )
        print(f"      FPS: {metadata['fps_original']} → {metadata['fps_new']}")
        print(f"      Resolution: {metadata['resolution_original']} → {metadata['resolution_new']}")
        print(f"      Frames: {metadata['frames_original']} → {metadata['frames_saved']}")

    except Exception as e:
        return {
            "processed_id": None,
            "signature_shape": None,
            "status"      : "error",
            "message"     : f"Error in preprocessing: {e}"
        }

    # --- STEP 2: Register in registry.csv and obtain ID --- #
    print("  [2/5] Registering in registry.csv...")

    processed_id = register_video(
        data_dir             = config["data_dir"],
        original_filename    = original_filename,
        preprocessing_metadata = metadata,
        notes                = notes,
    )
    print(f"      Assigned ID: {processed_id}")

    final_processed_path = os.path.join(
        config["processed_dir"],
        f"{processed_id}.mp4"
    )
    os.rename(temp_processed_path, final_processed_path)

    # --- STEP 3 & 4: Extract frames + landmarks --- #
    print("  [3/5] Extracting frames and landmarks...")

    try:
        landmarks_sequence = extract_video_landmarks(
            video_path       = final_processed_path,
            detector         = detector,
            similarity_threshold = config["ssim_threshold"],
        )
    except Exception as e:
        return {
            "processed_id": processed_id,
            "signature_shape": None,
            "status"      : "error",
            "message"     : f"Error in landmark extraction: {e}"
        }

    if not landmarks_sequence:
        return {
            "processed_id": processed_id,
            "signature_shape": None,
            "status"      : "error",
            "message"     : "No hands detected in any frame of the video."
        }

    print(f"      Frames with detected hands: {len(landmarks_sequence)}")

    # --- STEP 5: Build temporal signature --- #
    print("  [4/5] Building temporal signature...")

    try:
        signature        = build_signature(landmarks_sequence)
        signature_norm   = normalize_signature(signature)
        signature_smooth = smooth_signature(signature_norm, sigma=config["smoothing_sigma"])
    except Exception as e:
        return {
            "processed_id": processed_id,
            "signature_shape": None,
            "status"      : "error",
            "message"     : f"Error building signature: {e}"
        }

    print(f"      Signature shape: {signature_smooth.shape}")

    # --- STEP 6: Save signature .npy --- #
    print("  [5/5] Saving signature .npy...")

    signature_path = os.path.join(
        config["signatures_dir"],
        f"{processed_id}.npy"
    )
    np.save(signature_path, signature_smooth)
    mark_signature_saved(config["data_dir"], processed_id)

    print(f"      Signature saved at: {signature_path}")
    print(f"  ✅ {processed_id} processed successfully.")

    return {
        "processed_id": processed_id,
        "signature_shape": signature_smooth.shape,
        "status"      : "ok",
        "message"     : "Processing completed without errors."
    }

# ------------------------------------------------------------------- #
#                          Complete Pipeline                          #
# ------------------------------------------------------------------- #
def process_all_videos(config: dict = CONFIG) -> list[dict]:
    """
    Processes all .mp4 videos in raw/ that do not yet have a generated signature.
    Automatically detects which ones were already processed by comparing
    filenames in raw/ against registry.csv.

    Parameters:
        config : configuration dictionary

    Returns:
        List of results, one per processed video
    """
    existing_records = load_registry(config["data_dir"])
    already_processed = {r["original_filename"] for r in existing_records}

    all_videos = [
        f for f in os.listdir(config["raw_dir"])
        if f.lower().endswith(".mp4")
    ]
    pending = [f for f in all_videos if f not in already_processed]

    if not pending:
        print("✅ No new videos to process.")
        return []

    print(f"New videos found: {len(pending)} of {len(all_videos)} total")

    print(f"\nInitializing MediaPipe detector ({config['model_path']})...")
    detector = create_detector(
        model_path = config["model_path"],
        num_hands  = config["num_hands"],
    )

    results = []
    errors  = []

    for filename in sorted(pending):
        full_path = os.path.join(config["raw_dir"], filename)
        result    = process_single_video(full_path, detector, config)
        results.append(result)

        if result["status"] == "error":
            errors.append(result)

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Videos processed : {len(results)}")
    print(f"  Successful       : {len(results) - len(errors)}")
    print(f"  With errors      : {len(errors)}")

    if errors:
        print("\n  Videos with errors:")
        for e in errors:
            print(f"    - {e['processed_id']} : {e['message']}")

    return results

# ------------------------------------------------------------------- #
#                            Testing Area                             #
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    results = process_single_video("data/raw/Lavadodemanos-ReginaAllerAnton2.mp4", CONFIG)