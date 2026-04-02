"""
Hand landmark extraction module.
It is responsible for: 
    - Selecting distinctive frames using SSIM
    - Detecting landmarks with MediaPipe
    - Handling frames where hands are not detected (padding/interpolation)
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.metrics import structural_similarity as ssim

# ------------------------------------------------------------------- #
#    Detector initialization (it is created only once and reused)     #
# ------------------------------------------------------------------- #
def create_detector(model_path: str = "hand_landmarker.task", num_hands: int = 2):
    """
    Create and returns a MediaPipe HandLandmarker.

    Parameters:
        model_path : path to the models .task file
        num_hands  : maximum number of hands to detect

    Returns:
        vision.HandLandmarker 
    """
    base_options = python.BaseOptions(model_asset_path = model_path)
    options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = num_hands)

    return vision.HandLandmarker.create_from_options(options)

# ------------------------------------------------------------------- #
#                    Selecting distinctive frames                     #
# ------------------------------------------------------------------- #
def extract_distinctive_frames(
    video_path: str, 
    similarity: float = 0.90, 
    resize: tuple = (320, 240),
) -> list:
    """
    Extract distinctive frames based on structural similarity (SSIM).
    Only keep frames that differ enough from the previous one.

    Parameters:
        video_path : path to the video (already preprocessed)
        similarity : similar frames are discarded
        resize     : internal size for calculating SSIM (faster)

    Return: 
        list of BGR frames (numpy arrays) considered distinctive
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"The video could not be opened: {video_path}")
    
    distinctive_frames = []
    prev_frame_small   = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_original = frame.copy()
        frame_small    = cv2.resize(frame, resize)
        frame_gray     = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if prev_frame_small is None:
            distinctive_frames.append(frame_original)
            prev_frame_small = frame_gray
            continue

        score, _ = ssim(prev_frame_small, frame_gray, full = True)

        if score < similarity:
            distinctive_frames.append(frame_original)
            prev_frame_small = frame_gray

    cap.release()

    return distinctive_frames

# ------------------------------------------------------------------- #
#                 OpenCV frame conversion to mp.Image                 #
# ------------------------------------------------------------------- #
def openCVframe_to_mpImage(frame: np.ndarray) -> mp.Image:
    """
    Convert an OpenCV BGR frame into an mp.Image (RGB) object.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)

# ------------------------------------------------------------------- #
#            Landmark extraction with handling of absences            #
# ------------------------------------------------------------------- #
NUM_LANDMARKS = 21
NUM_COORDS = 3
NUM_HANDS = 2

# --- Landmarks block size by hand: 21 points × 3 coords --- #
LANDMARKS_BY_HAND = NUM_LANDMARKS * NUM_COORDS

# --- Auxiliar function to detect large nan gap --- #
def has_large_nan_gap(serie: np.ndarray, max_gap: int = 10) -> bool:
    """
    Check if there is a consecutive block of NaNs.
    """
    nan_mask = np.isnan(serie)
    consecutive_nans = 0

    for is_nan in nan_mask:
        if is_nan:
            consecutive_nans += 1
            if consecutive_nans > max_gap:
                return True
        else:
            consecutive_nans = 0
    
    return False

def extract_frame_landmarks(detection_result) -> dict:
    """
    Extract the world landmarks (hand_world_landmarks) from a frame and
    returns them as a fixed array (2, 21, 3).
    If a hand is not detected, its block is filled with NaN. This allows
    interpolation between frames later instead of using 0.

    Parameters: 
        detection_result : HandLandmarkerResult from MediaPipe

    Returns:
        np.ndarray (2, 21, 3) -> Always the same shape
        [0] = left hand (or NaN if not detected)
        [1] = right hand (or NaN if not detected)
        Normalized landmarks for visualization
    """
    exact_world = np.full((NUM_HANDS, NUM_LANDMARKS, NUM_COORDS), np.nan)
    exact_norm = np.full((NUM_HANDS, NUM_LANDMARKS, NUM_COORDS), np.nan)

    if detection_result is None or not detection_result.hand_world_landmarks:
        return {"world": exact_world, "normalized": exact_norm}
    
    # --- Iterate over both sets of reference points --- #
    zipped_data = zip(
        detection_result.hand_world_landmarks,
        detection_result.hand_landmarks,
        detection_result.handedness
    )

    for idx, (hand_world, hand_norm, handedness) in enumerate(zipped_data):
        if idx >= NUM_HANDS:
            break

        # --- MediaPipe returns "Left" or "Right" --- #
        side = handedness[0].category_name.lower()
        slot = 0 if side == "left" else 1

        exact_world[slot] = np.array([[lm.x, lm.y, lm.z] for lm in hand_world])
        exact_norm[slot] = np.array([[lm.x, lm.y, lm.z] for lm in hand_norm])

    return {"world": exact_world, "normalized": exact_norm}

def interpolate_landmarks(sequence: list[np.ndarray], max_gap: int = 10) -> list[np.ndarray]:
    """
    Receives a list of arrays (2, 21, 3) where some values are NaN and
    linearly interpolates the NaNs using the neighboring frames.
    If the first or last frame has NaN, it is filled with the nearest neighbor
    (forward/backward fill). If all the frames of a slot are NaN, that slot 
    remains 0 (the hand never appeared)

    Parameters: 
        secuence : list of np.ndarray (2, 21, 3)

    Returns:
        list of np.ndarray (2, 21, 3) without NaN 
    """
    T = len(sequence)

    # --- Transform to array (T, 2, 21, 3) --- #
    arr = np.array(sequence, dtype = np.float32)

    for hand_idx in range(NUM_HANDS):
        for lm_idx in range(NUM_LANDMARKS):
            for coord_idx in range(NUM_COORDS):
                serie = arr[:, hand_idx, lm_idx, coord_idx]

                nan_mask = np.isnan(serie)
                if not nan_mask.any():
                    continue

                if nan_mask.all():
                    # --- Hand never detected (filled with 0) --- #
                    arr[:, hand_idx, lm_idx, coord_idx] = 0.0
                    continue

                # --- Gap verification --- #
                if has_large_nan_gap(serie, max_gap):
                    raise ValueError(f"Gap greater than {max_gap} frames lost in hand {hand_idx}")

                # --- Linear interpolation --- #
                idx = np.arange(T)
                valids = ~nan_mask
                interpolated_serie = np.interp(idx, idx[valids], serie[valids])
                arr[:, hand_idx, lm_idx, coord_idx] = interpolated_serie

    return [arr[t] for t in range(T)]

# ------------------------------------------------------------------- #
#           Main function: video -> clean landmarks sequence          #
# ------------------------------------------------------------------- #
def extract_video_landmarks(
    video_path: str,
    detector,
    similarity: float = 0.88,
    max_gap: int = 10
) -> dict:
    """
    Complete pipeline: video -> list of landmarks per frame (without NaNs).

    Steps:
        1. Extract distinctive frames (SSIM)
        2. Detect landmarks with MediaPipe
        3. Interpolate missing landmarks

    Parameters:
        video_path : path of the preprocessed video
        detector   : HandLandmarker already initialized
        similarity : for SSIM filtering

    Returns:
        list of np.ndarray (2, 21, 3), one per distinctive frame
        empty list if no hands were detected in any frame
    """
    frames = extract_distinctive_frames(video_path, similarity)

    if not frames:
        return {"world": [], "normalized": []}
    
    raw_sequence = []
    for frame in frames:
        mp_image = openCVframe_to_mpImage(frame)
        result = detector.detect(mp_image)
        lm_frame = extract_frame_landmarks(result)
        raw_sequence.append(lm_frame)

    # --- Discard frames where no hand was detected (both slots NaN) --- #
    filtered_sequence = [lm for lm in raw_sequence if not np.isnan(lm["world"]).all()]

    if not filtered_sequence:
        return {"world": [], "normalized": []}
    
    # --- Separate data sets --- #
    world_sequence = [item["world"] for item in filtered_sequence]
    normalized_sequence = [item["normalized"] for item in filtered_sequence]
    
    # --- Interpolate missing landmarks --- #
    clean_world = interpolate_landmarks(world_sequence, max_gap)
    clean_normalized = interpolate_landmarks(normalized_sequence, max_gap)

    return {
        "world": clean_world,
        "normalized": clean_normalized,
        "frames": frames
    }