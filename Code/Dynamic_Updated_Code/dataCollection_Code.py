# fixed_100_frame_collector.py
# Collects fixed-length (100 frame) sign videos with clear phases:
# NEUTRAL -> SIGN NOW -> NEUTRAL AGAIN
# Updated: Safe indexing to avoid overwriting when some files are deleted

import cv2
import os
import re

# ------------------- CONFIG -------------------
DATA_PATH = '/Users/nahidkhan/Local Drive/Research/Dynamic Dataset'

# Countdown BEFORE each video starts (in seconds)
COUNTDOWN_SECONDS = 3

# Total frames per video (≈ time window)
FRAMES_PER_VIDEO = 100  # ~5 seconds if FPS ≈ 20

# Phase split: NEUTRAL -> SIGN -> NEUTRAL
NEUTRAL_START_FRAMES = 20   # প্রথম ২০ ফ্রেম neutral
SIGN_FRAMES          = 60   # মাঝের ৬০ ফ্রেমে sign
NEUTRAL_END_FRAMES   = 20   # শেষের ২০ ফ্রেম neutral/hold

# Safety check
assert NEUTRAL_START_FRAMES + SIGN_FRAMES + NEUTRAL_END_FRAMES == FRAMES_PER_VIDEO, \
    "Phase frame counts must sum to FRAMES_PER_VIDEO"
# ------------------------------------------------


def get_next_index(files, sign_name):
    """
    Find the max index from existing files like:
    sign_name_001.mp4, sign_name_002.mp4, ...
    Return max_index + 1 so we never overwrite.
    """
    pattern = re.compile(
        rf"^{re.escape(sign_name)}_(\d+)\.(mp4|avi|mov)$",
        re.IGNORECASE
    )
    max_idx = 0
    for f in files:
        m = pattern.match(f)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def main():
    # Create main data folder if not exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # ---- User inputs ----
    sign_name = input("Enter the dynamic sign name (label): ").strip()
    if not sign_name:
        print("Error: Sign name cannot be empty.")
        return

    try:
        total_videos_to_collect = int(
            input(f"How many NEW videos to collect for '{sign_name}': ")
        )
    except ValueError:
        print("Error: Please enter a valid integer for number of videos.")
        return

    # Per-sign folder
    sign_path = os.path.join(DATA_PATH, sign_name)
    if not os.path.exists(sign_path):
        os.makedirs(sign_path)

    # ---- Open webcam ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width == 0 or frame_height == 0:
        # Fallback resolution if camera doesn't report properly
        frame_width, frame_height = 640, 480

    print("\nPosition yourself in front of the camera.")
    print("Press 's' ONCE to start the entire batch recording, or 'q' to quit.\n")

    # ---- Wait until user presses 's' to start batch ----
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            cap.release()
            cv2.destroyAllWindows()
            return

        msg = "Press 's' to START batch | 'q' to quit"
        cv2.putText(frame, msg, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("100-Frame Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('s'):
            break

    # ---- SAFE starting index (no overwrite even if files deleted) ----
    existing_files = [
        f for f in os.listdir(sign_path)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    start_num = get_next_index(existing_files, sign_name)
    end_num = start_num + total_videos_to_collect - 1

    stop_all = False

    # ================= MAIN LOOP: per video =================
    for video_num in range(start_num, end_num + 1):
        if stop_all:
            break

        print(f"\nGet ready for video #{video_num} / {end_num}.")

        # ---- Visible countdown BEFORE recording ----
        for i in range(COUNTDOWN_SECONDS, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame during countdown.")
                stop_all = True
                break

            # Big countdown number in the center
            text = str(i)
            font_scale = 4
            thickness = 6
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_x = (frame_width - text_w) // 2
            text_y = (frame_height + text_h) // 2

            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            cv2.putText(frame, f"Sign: {sign_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Upcoming video #{video_num}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("100-Frame Capture", frame)
            key = cv2.waitKey(1000) & 0xFF   # 1 second per countdown step
            if key == ord('q'):
                stop_all = True
                break

        if stop_all:
            break

        # ---- Start recording exactly FRAMES_PER_VIDEO frames ----
        filename = os.path.join(sign_path, f"{sign_name}_{video_num:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'avc1' can also be used if supported
        out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

        print(f"Recording video #{video_num} --> {filename}")

        for frame_idx in range(1, FRAMES_PER_VIDEO + 1):
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame during recording.")
                stop_all = True
                break

            # Decide which phase we are in
            if frame_idx <= NEUTRAL_START_FRAMES:
                phase = "NEUTRAL"
            elif frame_idx <= NEUTRAL_START_FRAMES + SIGN_FRAMES:
                phase = "SIGN NOW"
            else:
                phase = "NEUTRAL AGAIN"

            # Overlays on frame
            cv2.putText(frame, f"Sign: {sign_name}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Video #{video_num} / {end_num}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame {frame_idx} / {FRAMES_PER_VIDEO}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, phase, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to stop all", (20, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            out.write(frame)
            cv2.imshow("100-Frame Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_all = True
                break

        out.release()

        if stop_all:
            print("Stopping early due to 'q' key.")
            break

        print(f"Saved: {filename}")

    print("\nBatch recording complete!")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
