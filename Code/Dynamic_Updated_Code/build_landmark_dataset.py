# build_landmark_dataset.py
import os, cv2, numpy as np, mediapipe as mp

DATA_PATH = '/Users/nahidkhan/Local Drive/Research/Dynamic Dataset'   # <-- তোমার dataset folder path
OUT_PATH = '/Users/nahidkhan/Local Drive/Research/Landmark_Data'   # output folder
T = 100                      # frames per video
USE_MIDDLE = True            # middle 60 frames also save

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

def extract_one_video(video_path, T=100):
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

def smooth_sequence(seq, k=3):
    if k <= 1: return seq
    out = seq.copy()
    for t in range(len(seq)):
        s = max(0, t-k//2); e = min(len(seq), t+k//2+1)
        out[t] = seq[s:e].mean(axis=0)
    return out

def main():
    os.makedirs(OUT_PATH, exist_ok=True)

    classes = sorted([d for d in os.listdir(DATA_PATH)
                      if os.path.isdir(os.path.join(DATA_PATH, d))])
    label_to_id = {c:i for i,c in enumerate(classes)}

    X100, X60, y = [], [], []

    print("Classes:", label_to_id)

    for c in classes:
        folder = os.path.join(DATA_PATH, c)
        vids = sorted([f for f in os.listdir(folder)
                       if f.lower().endswith((".mp4",".avi",".mov"))])

        print(f"\nProcessing {c} ({len(vids)} videos)...")

        for vf in vids:
            vp = os.path.join(folder, vf)
            seq100 = extract_one_video(vp, T=T)     # (100,126)
            seq100 = smooth_sequence(seq100, k=3)   # optional smoothing

            X100.append(seq100)
            y.append(label_to_id[c])

            if USE_MIDDLE:
                seq60 = seq100[20:80]              # middle 60 frames
                X60.append(seq60)

    X100 = np.stack(X100).astype(np.float32)  # (510,100,126)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(OUT_PATH, "X100.npy"), X100)
    np.save(os.path.join(OUT_PATH, "y.npy"), y)
    np.save(os.path.join(OUT_PATH, "label_names.npy"), np.array(classes))

    print("\nSaved X100.npy, y.npy, label_names.npy")
    print("X100 shape:", X100.shape, "y shape:", y.shape)

    if USE_MIDDLE:
        X60 = np.stack(X60).astype(np.float32)    # (510,60,126)
        np.save(os.path.join(OUT_PATH, "X60.npy"), X60)
        print("Saved X60.npy")
        print("X60 shape:", X60.shape)

if __name__ == "__main__":
    main()
