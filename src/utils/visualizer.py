import cv2
import numpy as np

def create_debug_video(
    frames: list[np.ndarray],
    landmarks_seq: list[np.ndarray],
    output_path: str,
    fps: int = 15
) -> None:
    """
    Take the distinctive frames and the normalized landmarks, draw
    the points, and save the result in a temporary video file.
    """
    if not frames or not landmarks_seq:
        print("No frames or landmarks to visualize.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame, landmarks in zip(frames, landmarks_seq):
        debug_frame = frame.copy()

        for hand_idx in range(2):
            hand_pts = landmarks[hand_idx]
            
            if np.all(hand_pts == 0.0):
                continue

            for pt in hand_pts:
                x = int(pt[0] * width)
                y = int(pt[1] * height)
                
                color = (0, 255, 0) if hand_idx == 0 else (255, 0, 0)
                cv2.circle(debug_frame, (x, y), 5, color, -1)

        out.write(debug_frame)

    out.release()
    print(f"      Video saved in: {output_path}")