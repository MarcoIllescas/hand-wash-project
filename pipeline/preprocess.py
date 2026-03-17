"""
Video standarization module.
It is responsible for:
    - Temporal rescaling (fixed FPS)
    - Spatial normalization (resize)
    - Lighting correction (CLAHE)
"""
import cv2
import numpy as np

def preprocess_video(
    input_path: str,
    output_path: str,
    target_fps: int = 25,
    target_size: tuple = (640, 480),
    clahe_application: bool = True
) -> dict:
    """
    Preprocess a video by performing:
        Temporal rescaling (fixed FPS)
        Spatial normalization (resize)
        Basic lighting correction (CLAHE)

    Parameters:
        input_path        : path of the original video
        output_path       : path of the processed video
        target_fps        : desired FPS (default 25)
        target_size       : desired (width, height) (default 640x480)
        clahe_application : apply lighting correction

    Returns:
        dict with processing metadata

    Raises:
        ValueError if the video cannot be opened
        RuntimeError if the VideoWriter cannot be created
    """

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"The video could not be opened: {input_path}")
    
    original_fps          = cap.get(cv2.CAP_PROP_FPS)
    original_width        = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height       = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    original_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # --- Avoid division by 0 if the video reports FPS = 0 --- #
    if original_fps <= 0:
        cap.release()
        raise ValueError(f"The video has an invalid FPS ({original_fps}): {input_path}")
    
    # --- Temporary sampling factor --- #
    frame_interval = original_fps / target_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, target_size)

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"The output file could not be created: {output_path}")
    
    if clahe_application:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    frame_idx       = 0
    saved_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Temporal subsampling --- #
        if frame_idx >= (saved_frame_idx * frame_interval):
            # --- Spatial rezise --- #
            frame = cv2.resize(frame, target_size)

            # --- Lighting correction in LAB space --- #
            if clahe_application: 
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            out.write(frame)
            saved_frame_idx += 1
        
        frame_idx += 1

    cap.release()
    out.release()

    return {
        "original_fps"        : original_fps,
        "new_fps"             : target_fps,
        "original_resolution" : (original_width, original_height),
        "new_resolution"      : target_size,
        "original_frames"     : original_total_frames,
        "saved_frames"        : saved_frame_idx,
    }