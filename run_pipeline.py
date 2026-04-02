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
from src.signature_engine import SignatureEngine
from src.utils.registry_manager import load_registry

# ------------------------------------------------------------------- #
#                        General configuration                        #
# ------------------------------------------------------------------- #
CONFIG = {
    # Paths
    "raw_dir"        : "data/raw",
    "processed_dir"  : "data/processed",
    "signatures_dir" : "data/signatures",
    "debug_dir"      : "data/debug",
    "data_dir"       : "data",

    # Preprocessing
    "target_fps"     : 25,
    "target_size"    : (640, 480),
    "apply_clahe"    : True,

    # Landmark extraction
    "model_path"     : "hand_landmarker.task",
    "num_hands"      : 2,
    "ssim_threshold" : 0.88,
    "max_gap"        : 100,

    # Signature
    "smoothing_sigma": 1.0,
}

# ------------------------------------------------------------------- #
#                       Pipeline Test (1 video)                       #
# ------------------------------------------------------------------- #
def process_pending_videos(engine: SignatureEngine):
    """
    Scan data/raw dir, check registry and process only new data.
    """
    if not os.path.exists(CONFIG["raw_dir"]):
        print(f"Dir {CONFIG['raw_dir']} does not exist.")
        return
    
    # --- Load registry to know what videos must skip --- #
    existing_records = load_registry(CONFIG["data_dir"])
    already_processed = {r["original_filename"] for r in existing_records}

    # --- Read all videos (support mp4, avi, mov) --- #
    all_videos = [f for f in os.listdir(CONFIG["raw_dir"]) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    pending = [f for f in all_videos if f not in already_processed]

    if not pending:
        print("There is no new videos to precess.")
        return
    
    print(f"Found {len(pending)} new video(s) out of a total {len(all_videos)}.")

    results = []
    errors = []

    for filename in sorted(pending):
        result = engine.process_single_video(filename)
        results.append(result)

        if result["status"] == "error":
            errors.append((filename, result["message"]))

    print(f"\n{'='*50}")
    print("  FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"  Processed videos  : {len(results)}")
    print(f"  Success           : {len(results) - len(errors)}")
    print(f"  Errores           : {len(errors)}")

    if errors:
        print("\n  Error details:")
        for file, error in errors:
            print(f"    - {file}: {error}")

# ------------------------------------------------------------------- #
#                               Execution                             #
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Starting HandWash Pipeline.\n")

    engine = SignatureEngine(CONFIG)

    # --- OPTION 1: Process a single video --- #
    result = engine.process_single_video("video_reference.mp4")
    print(result)

    # --- OPTION 2: Process all pending videos --- #
    # process_pending_videos(engine)