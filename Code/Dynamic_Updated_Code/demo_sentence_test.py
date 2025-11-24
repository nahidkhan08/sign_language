import numpy as np
import tensorflow as tf
from sentence_generator import generate_sentence

# Load
X_test = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_test.npy").astype(np.float32)
y_test = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_test.npy").astype(np.int64)
label_names = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy", allow_pickle=True)

model = tf.keras.models.load_model("/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm_tf.keras")

# Predict labels
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)

# Convert to words
pred_words = [label_names[i] for i in y_pred]
true_words = [label_names[i] for i in y_test]

print("Some predictions:")
for i in range(10):
    print(f"{i+1:02d}. true={true_words[i]}  pred={pred_words[i]}")

# ---- Simple 2-word sentence demo ----
# For demo: take consecutive predictions as if user signed two words in a row
print("\nSentence demo from consecutive predictions:")
for i in range(0, 10, 2):
    seq = [pred_words[i], pred_words[i+1]]
    sent = generate_sentence(seq)
    print(seq, "->", sent)
