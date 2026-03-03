def preprocess_video(input_path, output_path, target_fps=25, target_size=(640, 480), apply_clahe=True):
    """
    Preprocess a video following the next steps:
        1. Temporal Rescaling (25 FPS)
        2. Spatial Rescaling (640x480)
        3. Illumination Correction (CLAHE)
    
    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the preprocessed video.
        target_fps (int): Target frames per second.
        target_size (tuple): Target size (width, height).
        apply_clahe (bool): Whether to apply CLAHE.

    Returns:
        dict: metadata of processed video
    """
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video: {original_fps} FPS, {original_width}x{original_height}, {original_total_frames} frames")