import os
import numpy as np
from src.pipeline.preprocessor import preprocess_video
from src.pipeline.extractor import create_detector, extract_video_landmarks
from src.pipeline.builder import build_signature, normalize_signature, smooth_signature
from src.utils.registry_manager import register_video, mark_signature_saved, load_registry
from src.utils.visualizer import create_debug_video

class SignatureEngine:
    def __init__(self, config: dict):
        """
        Initializes the engine by loading the configuration and the MediaPipe model.
        """
        self.config = config
        print(f"Initializing MediaPipe model from: {config['model_path']}")
        self.detector = create_detector(
            model_path = config["model_path"], 
            num_hands = config["num_hands"]
        )

    def process_single_video(self, original_filename: str, debug: bool = False) -> dict:
        """
        Runs the complete pipeline for a single video.
        """
        raw_path = os.path.join(self.config["raw_dir"], original_filename)
        
        print(f"\n{'-'*50}\nProcessing: {original_filename}\n{'-'*50}")

        # --- STEP 1: Preprocessing --- #
        temp_processed_path = os.path.join(self.config["processed_dir"], f"temp_{original_filename}")
        
        try:
            print(" [1/6] Standardizing video.")
            metadata = preprocess_video(
                input_path = raw_path,
                output_path = temp_processed_path,
                target_fps = self.config["target_fps"],
                target_size = self.config["target_size"],
                clahe_application = self.config["apply_clahe"]
            )
        except Exception as e:
            return {"status": "error", "message": f"Preprocessing failed: {e}"}

        # --- STEP 2: Register in CSV to obtain the ID --- #
        print(" [2/6] Generating ID and registering.")
        processed_id = register_video(
            data_dir = self.config["data_dir"],
            original_filename = original_filename,
            preprocessing_metadata = metadata
        )

        final_precessed_path = os.path.join(self.config["processed_dir"], f"{processed_id}.mp4")
        os.rename(temp_processed_path, final_precessed_path)

        # --- STEP 3: Landmark Extraction --- #
        print(" [3/6] Extracting distinctive frames and landmarks.")
        try:
            extracted_data = extract_video_landmarks(
                video_path = final_precessed_path,
                detector = self.detector,
                similarity = self.config["ssim_threshold"],
                max_gap = self.config.get("max_gap", 5)
            )
        except Exception as e:
            return {"status": "error", "message": f"Extraction failed: {e}"}

        if not extracted_data["world"]:
            return {"status": "error", "message": "No hands were detected in the video."}

        # --- STEP 4: Generate Debug Video --- #
        print(" [4/6] Generating Video Debug.")
        debug_path = os.path.join(self.config["debug_dir"], f"debug_{processed_id}.mp4")
        create_debug_video(
            frames = extracted_data["frames"],
            landmarks_seq = extracted_data["normalized"],
            output_path = debug_path
        )

        # --- STEP 5: Build the Signature --- #
        print(" [5/6] Building vector signature.")
        try:
            signature = build_signature(extracted_data["world"])
            signature_norm = normalize_signature(signature)
            signature_smooth = smooth_signature(signature_norm, sigma=self.config["smoothing_sigma"])
        except Exception as e:
            return {"status": "error", "message": f"Signature construction failed: {e}"}

        # --- STEP 6: Save .npy --- #
        print(" [6/6] Saving signature.")
        signature_path = os.path.join(self.config["signatures_dir"], f"{processed_id}.npy")
        np.save(signature_path, signature_smooth)
        
        mark_signature_saved(self.config["data_dir"], processed_id)

        print(f" ✅ Process successful! Signature {processed_id} generated.")
        return {"status": "ok", "processed_id": processed_id}
