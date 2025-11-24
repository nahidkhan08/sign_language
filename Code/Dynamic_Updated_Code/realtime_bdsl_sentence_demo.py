# realtime_bdsl_sentence_demo.py
# Live BdSL Word Recognition + Simple Sentence Generator
# Controls:
#   SPACE = Start/Stop detection
#   C     = Clear accepted words / reset sentence
#   Q     = Quit

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from sentence_generator import generate_sentence

# --- NEW: PIL for Bangla text ---
from PIL import ImageFont, ImageDraw, Image

# ------------------------ CONFIG ------------------------
MODEL_PATH = "/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm_tf.keras"
LABEL_PATH = "/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy"

# --- NEW: Bangla font path (update if needed) ---
FONT_PATH = '/Users/nahidkhan/Local Drive/Research/fonts/NotoSansBengali_Condensed-Medium.ttf'
BN_FONT_SIZE = 38  # adjust if too big/small

T = 100
STABLE_N = 3
CONF_THRESH = 0.6
COOLDOWN_SECONDS = 1.0
FPS_ASSUMED = 20
MAX_WORDS_SHOW = 6
# --------------------------------------------------------

# -------------------- Load model/labels -----------------
model = tf.keras.models.load_model(MODEL_PATH)
label_names = np.load(LABEL_PATH, allow_pickle=True)

# -------------------- Load Bangla font -----------------
bn_font = ImageFont.truetype(FONT_PATH, BN_FONT_SIZE)

def put_bangla_text(frame, text, pos, font=bn_font, color=(0, 0, 0)):
    """
    Draw Bangla text on OpenCV frame using PIL.
    color is RGB (not BGR).
    """
    if text is None:
        text = ""
    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    draw.text(pos, text, font=font, fill=color)

    # RGB -> BGR
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame

# -------------------- MediaPipe init --------------------
mp_hands = mp.solutions.hands

# -------------------- Helper functions -----------------
def normalize_hand(hand21x3):
    if np.all(hand21x3 == 0):
        return hand21x3
    wrist = hand21x3[0].copy()
    hand = hand21x3 - wrist
    palm_size = np.linalg.norm(hand21x3[9] - hand21x3[0]) + 1e-6
    return hand / palm_size


def extract_frame_features(frame, hands):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, info in zip(res.multi_hand_landmarks, res.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            label = info.classification[0].label
            if label == "Left":
                left = coords
            else:
                right = coords

    left = normalize_hand(left)
    right = normalize_hand(right)

    return np.concatenate([left.flatten(), right.flatten()])  # (126,)


def draw_ui(frame, pred_word, pred_conf, accepted_words, sentence,
            running, cooldown_left, phase_text):
    h, w, _ = frame.shape

    status_text = "RUNNING" if running else "PAUSED"
    status_color = (0, 255, 0) if running else (0, 0, 255)

    # Status
    cv2.putText(frame, f"Status: {status_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Phase / Instruction
    cv2.putText(frame, f"{phase_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Prediction
    cv2.putText(frame, f"Pred: {pred_word}  (conf {pred_conf:.2f})", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Words list
    show_words = accepted_words[-MAX_WORDS_SHOW:]
    cv2.putText(frame, f"Words: {show_words}", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Sentence label (English) with OpenCV
    cv2.putText(frame, "Sentence:", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Sentence (Bangla) with PIL
    # Position tuned to appear next to "Sentence:"
    frame = put_bangla_text(frame, sentence, (170, 185), font=bn_font, color=(0, 0, 0))

    # Cooldown
    if cooldown_left > 0:
        cv2.putText(frame, f"Cooldown: {cooldown_left:.1f}s", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Help
    cv2.putText(frame, "SPACE: Start/Stop | C: Clear | Q: Quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    return frame


# -------------------------- Main -------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    buffer = deque(maxlen=T)

    accepted_words = []
    last_pred = None
    stable_count = 0
    running = False

    cooldown_frames_total = int(COOLDOWN_SECONDS * FPS_ASSUMED)
    cooldown_frames_left = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #frame = cv2.flip(frame, 1)

            pred_word = ""
            pred_conf = 0.0

            phase_text = "PAUSED"

            if running:
                feat = extract_frame_features(frame, hands)
                buffer.append(feat)

                if cooldown_frames_left > 0:
                    cooldown_frames_left -= 1

                # Phase logic
                if len(buffer) < T:
                    phase_text = "GET READY (Hold Neutral)"
                elif cooldown_frames_left > 0:
                    phase_text = "WAIT (Hold Neutral)"
                else:
                    phase_text = "SIGN NOW!"

                if len(buffer) == T:
                    seq = np.stack(buffer)[None, ...].astype(np.float32)

                    prob = model.predict(seq, verbose=0)[0]
                    pid = int(prob.argmax())
                    pred_conf = float(prob.max())
                    pred_word = str(label_names[pid])

                    # low confidence ignore
                    if pred_conf < CONF_THRESH:
                        pred_word = ""
                        if cooldown_frames_left == 0:
                            phase_text = "HOLD NEUTRAL"

                    # stability check only if not in cooldown and pred_word valid
                    if cooldown_frames_left == 0 and pred_word:
                        if pred_word == last_pred:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_pred = pred_word

                        if stable_count >= STABLE_N:
                            if len(accepted_words) == 0 or accepted_words[-1] != pred_word:
                                accepted_words.append(pred_word)
                                cooldown_frames_left = cooldown_frames_total
                                phase_text = "WAIT (Hold Neutral)"
                            stable_count = 0
                    else:
                        stable_count = 0
                        last_pred = pred_word

            # Sentence from last 2 accepted words
            sentence = generate_sentence(accepted_words[-2:]) if len(accepted_words) >= 2 else ""
            cooldown_left_sec = cooldown_frames_left / FPS_ASSUMED

            frame = draw_ui(frame, pred_word, pred_conf,
                            accepted_words, sentence, running,
                            cooldown_left_sec, phase_text)

            cv2.imshow("Live BdSL Sentence Demo", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                running = not running
                stable_count = 0
                last_pred = None
                print("RUNNING" if running else "PAUSED")
            elif key == ord('c'):
                accepted_words = []
                stable_count = 0
                last_pred = None
                cooldown_frames_left = 0
                print("Cleared words/sentence.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
