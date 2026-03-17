"""
Geometric feature construction module.
It receives the sequence of clean landmarks and produces a numerical 
vector per frame. 

Calculated features:
    1. Palm-to-palm distance
    2. Tip-to-tip distances (tip of each finger between hands)
    3. Joint angles (MCP, PIP, DIP) per finger
    4. Global orientation of each hand
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ------------------------------------------------------------------- #
#             Reference indexes (MediaPipe Hand Landmarks)            #
# ------------------------------------------------------------------- #
#                                                                     #
#    0   = WRIST                                                      #
#   1-4  = THUMB  (CMC, MCP, IP, TIP)                                 #
#   5-8  = INDEX  (MCP, PIP, DIP, TIP)                                #
#   9-12 = MIDDLE (MCP, PIP, DIP, TIP)                                #
#  13-16 = RING   (MCP, PIP, DIP, TIP)                                #
#  17-20 = PINKY  (MCP, PIP, DIP, TIP)                                #
# ------------------------------------------------------------------- #

PALM_CENTER_IDX = [0, 5, 9, 13, 17]
TIPS_IDX        = [4, 8, 12, 16, 20]

FINGERS = {
    "thumb"   : [1, 2, 3, 4],
    "index"   : [5,  6,  7,  8],
    "middle"  : [9,  10, 11, 12],
    "ring"    : [13, 14, 15, 16],
    "pinky"   : [17, 18, 19, 20],
}

# --- Fixed dimension of the feature vector per frame --- #
# 1 (palm-palm) + 5 (tip-tip) + 2 (hands) × 5 (fingers) × 3 (angles) + 2 (hands) × 6 (orientation) #
DIM_FEATURES = 48

# ------------------------------------------------------------------- #
#                          Auxiliar functions                         #
# ------------------------------------------------------------------- #
def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point b formed by the vectors b->a and b->c
    Returns degrees in [0, 180]
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))

# ------------------------------------------------------------------- #
#                          Feature per frame                          #
# ------------------------------------------------------------------- #
def calculate_features_frame(landmarks_frame: np.ndarray) -> np.ndarray:
    """
    Computes the feature vector for one frame.

    Parameters: 
        landmarks_frame : np.ndarray (2, 21, 3)
                          [0] = left hand
                          [1] = right hand
    
    Returns:
        1D np.ndarray of length DIM_FEATURES
    """
    l_hand = landmarks_frame[0]
    r_hand = landmarks_frame[1]
    vector = []

    # --- 1. Palm-to-palm distance --- #
    left_center = np.mean(l_hand[PALM_CENTER_IDX], axis = 0)
    right_center = np.mean(r_hand[PALM_CENTER_IDX], axis = 0)
    vector.append(_distance(left_center, right_center))

    # --- 2. Tip-to-tip distances (fingertip of each finger between hands) --- #
    for tip_idx in TIPS_IDX:
        vector.append(_distance(l_hand[tip_idx], r_hand[tip_idx]))

    # --- 3. Joint angles (MCP, PIP, DIP) per finger --- #
    for hand in [l_hand, r_hand]:
        for name, points in sorted(FINGERS.items()):
            # --- MCP: wrist -> MCP -> PIP --- #
            ang_mcp = _angle(hand[0], hand[points[0]], hand[points[1]])
            # --- PIP: MCP -> PIP -> DIP --- #
            ang_pip = _angle(hand[points[0]], hand[points[1]], hand[points[2]])
            # --- DIP: PIP -> DIP -> TIP --- #
            ang_dip = _angle(hand[points[1]], hand[points[2]], hand[points[3]])

            vector.extend([ang_mcp, ang_pip, ang_dip])

    # --- 4. Global orientation of each hand --- #
    for hand in [l_hand, r_hand]:
        # --- Main vector: wrist -> middle finger MCP
        v1 = hand[9] - hand[0]
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)

        # --- Transversal vector: index MCP -> pinky MCP
        v2 = hand[17] - hand[5]

        # --- Palm normal --- #
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        vector.extend(v1.tolist())
        vector.extend(normal.tolist())

    result = np.array(vector, dtype = np.float32)

    # --- Safety: If size does not match, pad or truncate --- #
    if len(result) < DIM_FEATURES:
        result = np.pad(result, (0, DIM_FEATURES - len(result)))
    elif len(result) > DIM_FEATURES:
        result = result[:DIM_FEATURES]

    return result

# ------------------------------------------------------------------- #
#           Construction of the complete temporal signature           #
# ------------------------------------------------------------------- #
def build_signature(landmarks_sequence: list[np.ndarray]) -> np.ndarray:
    """
    Converts a sequence of landmarks into the temporal signature of the 
    video.

    Parameters: 
        landmarks_sequence : list of np.ndarray (2, 21, 3), one per frame

    Returns:
        np.ndarray of shape (T, DIM_FEATURES)
        where T = number of distinctive frames
    """
    if not landmarks_sequence:
        raise ValueError("The landmarks sequence is empty.")
    
    frames_features = [calculate_features_frame(lm_frame) for lm_frame in landmarks_sequence]

    return np.vstack(frames_features)

# ------------------------------------------------------------------- #
#                     Normalization and smoothing                     #
# ------------------------------------------------------------------- #
def normalize_signature(signature: np.ndarray) -> np.ndarray:
    """
    Z-Score normalization per column (feature by feature)
    """
    mean = np.mean(signature, axis = 0)
    std = np.std(signature, axis = 0) + 1e-8

    return (signature - mean) / std

def smooth_signature(signature: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing along the temporal axis
    """
    return gaussian_filter1d(signature, sigma = sigma, axis = 0)