import os, cv2, numpy as np
import mediapipe as mp

DATA_PATH = '/Users/nahidkhan/Local Drive/Research/Sample'   # তোমার data folder
OUT_PATH = '/Users/nahidkhan/Local Drive/Research/Pilot Landmark'
T = 100  # frames per video

mp_hands = mp.solutions.hands

def normalize_hand(hand21x3):
    if np.all(hand21x3 == 0):
        return hand21x3
    wrist = hand21x3[0].copy()
    hand = hand21x3 - wrist
    palm_size = np.linalg.norm(hand21x3[9] - hand21x3[0]) + 1e-6
    return hand / palm_size

def pad_or_crop(seq, T):
    if len(seq) == 0:
        return [np.zeros((126,), dtype=np.float32)] * T
    if len(seq) >= T:
        start = (len(seq)-T)//2
        return seq[start:start+T]
    last = seq[-1]
    return seq + [last]*(T-len(seq))

def extract_one_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            left = np.zeros((21,3), dtype=np.float32)
            right = np.zeros((21,3), dtype=np.float32)

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, info in zip(res.multi_hand_landmarks, res.multi_handedness):
                    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    label = info.classification[0].label  # "Left"/"Right"
                    if label == "Left":
                        left = coords
                    else:
                        right = coords

            left = normalize_hand(left)
            right = normalize_hand(right)

            feat = np.concatenate([left.flatten(), right.flatten()])  # (126,)
            frames.append(feat)

    cap.release()
    frames = pad_or_crop(frames, T)
    return np.stack(frames)  # (100,126)

def main():
    os.makedirs(OUT_PATH, exist_ok=True)

    classes = sorted([d for d in os.listdir(DATA_PATH)
                      if os.path.isdir(os.path.join(DATA_PATH, d))])

    print("Classes found:", classes)

    for c in classes:
        in_folder = os.path.join(DATA_PATH, c)
        out_folder = os.path.join(OUT_PATH, c)
        os.makedirs(out_folder, exist_ok=True)

        vids = sorted([f for f in os.listdir(in_folder)
                       if f.lower().endswith(".mp4")])[:5]  # only 5 videos

        print(f"\nProcessing class {c}, {len(vids)} videos...")
        for vf in vids:
            vp = os.path.join(in_folder, vf)
            seq = extract_one_video(vp)
            save_name = os.path.splitext(vf)[0] + ".npy"
            np.save(os.path.join(out_folder, save_name), seq)
            print(" saved", save_name, seq.shape)

    print("\nPilot extraction done. Check Pilot_Landmarks folder.")

if __name__ == "__main__":
    main()
