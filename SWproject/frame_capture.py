import cv2
import os

SAVE_DIR = "detected_frames"

os.makedirs(SAVE_DIR, exist_ok=True)

saved_count = 0


def save_detected_frame(frame):
    global saved_count

    filename = os.path.join(
        SAVE_DIR,
        f"frame_{saved_count:06d}.jpg"
    )

    cv2.imwrite(filename, frame)

    saved_count += 1