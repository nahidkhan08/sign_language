import numpy as np
import tensorflow as tf
from sentence_generator import generate_sentence

# ------------ load model + labels ------------
model = tf.keras.models.load_model(
    "/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm_tf.keras"
)
label_names = np.load(
    "/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy",
    allow_pickle=True
)

# ------------ helper: majority vote smoothing ------------
def smooth_preds(pred_ids, k=3):
    smoothed = []
    for i in range(len(pred_ids)):
        s = max(0, i-k//2)
        e = min(len(pred_ids), i+k//2+1)
        window = pred_ids[s:e]
        smoothed.append(int(np.bincount(window).argmax()))
    return smoothed

# ------------ load landmark dataset ------------
X100 = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X100.npy")  # (515,100,126)
y = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y.npy")

def pick_one(class_name):
    idx = np.where(label_names[y] == class_name)[0][0]
    return X100[idx]

# --- pick two words to simulate continuous signing ---
seq1 = pick_one("Ami")   # subject
seq2 = pick_one("asa")   # verb

# ------------ build continuous stream ------------
stream = np.concatenate([seq1, seq2], axis=0)  # (200,126)

# ------------ sliding window prediction ------------
WINDOW = 100
STRIDE = 20
CONF_THRESH = 0.8   # <-- transition noise কাটার জন্য

pred_ids = []
pred_confs = []

for start in range(0, len(stream) - WINDOW + 1, STRIDE):
    win = stream[start:start + WINDOW][None, ...]   # (1,100,126)

    prob = model.predict(win, verbose=0)[0]  # (num_classes,)
    pid = int(prob.argmax())
    conf = float(prob.max())

    # Only keep high-confidence windows
    if conf >= CONF_THRESH:
        pred_ids.append(pid)
        pred_confs.append(conf)

print("Raw window preds (filtered):",
      [label_names[i] for i in pred_ids])
print("Raw confs:", [round(c, 3) for c in pred_confs])

# ------------ smoothing ------------
if len(pred_ids) == 0:
    print("\nNo windows passed confidence threshold. Lower CONF_THRESH (e.g., 0.7).")
    final_words = []
else:
    pred_ids_sm = smooth_preds(pred_ids, k=3)
    pred_words = [label_names[i] for i in pred_ids_sm]

    print("Smoothed preds:", pred_words)

    # ------------ take unique transitions ------------
    final_words = []
    for w in pred_words:
        if len(final_words) == 0 or final_words[-1] != w:
            final_words.append(w)

print("Final word sequence:", final_words)

# ------------ sentence generation ------------
sentence = generate_sentence(final_words)
print("Generated sentence:", sentence)
