import cv2, numpy as np, mediapipe as mp, tensorflow as tf
from collections import deque
from sentence_generator import generate_sentence

# ---------------- Load model + labels ----------------
model = tf.keras.models.load_model(
    "/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm_tf.keras"
)
label_names = np.load(
    "/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy",
    allow_pickle=True
)

mp_hands = mp.solutions.hands

# ---------------- Parameters ----------------
T = 100
buffer = deque(maxlen=T)

accepted_words = []
last_pred = None
stable_count = 0
STABLE_N = 3              # same pred 3 times -> accept
CONF_THRESH = 0.6         # Fix A: low confidence ignore

FPS_ASSUMED = 20
COOLDOWN_SECONDS = 1.2    # Fix B: word accept er por gap
COOLDOWN_FRAMES = int(COOLDOWN_SECONDS * FPS_ASSUMED)
cooldown_left = 0

MOTION_THRESH = 0.002     # Fix C: neutral/still detect threshold

# ---------------- Helpers ----------------
def normalize_hand(hand21x3):
    if np.all(hand21x3 == 0): 
        return hand21x3
    wrist = hand21x3[0].copy()
    hand = hand21x3 - wrist
    palm_size = np.linalg.norm(hand21x3[9] - hand21x3[0]) + 1e-6
    return hand / palm_size

def motion_energy(buf):
    """last 2 frame er feature difference er avg -> still/neutral detect"""
    if len(buf) < 2:
        return 999.0
    a = buf[-1]
    b = buf[-2]
    return float(np.mean(np.abs(a - b)))

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # Mirror view makes signing easier
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        left = np.zeros((21,3), np.float32)
        right = np.zeros((21,3), np.float32)

        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, info in zip(res.multi_hand_landmarks, res.multi_handedness):
                coords = np.array([[p.x,p.y,p.z] for p in lm.landmark], np.float32)
                if info.classification[0].label == "Left":
                    left = coords
                else:
                    right = coords

        left = normalize_hand(left)
        right = normalize_hand(right)

        feat = np.concatenate([left.flatten(), right.flatten()])  # (126,)
        buffer.append(feat)

        pred_word = ""
        pred_conf = 0.0

        if len(buffer) == T:
            seq = np.stack(buffer)[None, ...].astype(np.float32)  # (1,100,126)
            prob = model.predict(seq, verbose=0)[0]
            pid = int(prob.argmax())
            pred_conf = float(prob.max())

            # Fix A: confidence gate
            if pred_conf >= CONF_THRESH:
                pred_word = str(label_names[pid])
            else:
                pred_word = ""

            # Fix C: neutral/still gate (neutral hold e accept off)
            if motion_energy(buffer) < MOTION_THRESH:
                pred_word = ""
                stable_count = 0
                last_pred = None

            # Fix B: cooldown (word accept er por kichu frame ignore)
            if cooldown_left > 0:
                cooldown_left -= 1
                stable_count = 0  # cooldown e stability count reset
            else:
                # stability check only if pred_word valid
                if pred_word:
                    if pred_word == last_pred:
                        stable_count += 1
                    else:
                        stable_count = 1
                        last_pred = pred_word

                    # accept word
                    if stable_count >= STABLE_N:
                        if len(accepted_words)==0 or accepted_words[-1]!=pred_word:
                            accepted_words.append(pred_word)
                            cooldown_left = COOLDOWN_FRAMES  # start cooldown
                        stable_count = 0

        # sentence show
        sentence = generate_sentence(accepted_words[-2:]) if len(accepted_words)>=2 else ""

        # ---------------- UI overlay (all black text) ----------------
        cv2.putText(frame, f"Pred: {pred_word} (conf {pred_conf:.2f})", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        cv2.putText(frame, f"Words: {accepted_words[-4:]}", (20,80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        cv2.putText(frame, f"Sentence: {sentence}", (20,130),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        if cooldown_left > 0:
            cv2.putText(frame, f"Cooldown: {cooldown_left/FPS_ASSUMED:.1f}s", (20,170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.putText(frame, "Press 'c' to clear | 'q' to quit", (20, frame.shape[0]-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cv2.imshow("Live BdSL Sentence Demo (Fixed)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            accepted_words = []
            stable_count = 0
            last_pred = None
            cooldown_left = 0
            print("Cleared words.")

cap.release()
cv2.destroyAllWindows()
