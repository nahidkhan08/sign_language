# step4_eval.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -------- Load test data --------
X_test = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_test.npy").astype(np.float32)
y_test = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_test.npy").astype(np.int64)
label_names = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy", allow_pickle=True)

print("X_test:", X_test.shape, "y_test:", y_test.shape)
print("Labels:", label_names)

# -------- Load best model --------
model = tf.keras.models.load_model("/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm_tf.keras")

# -------- Predict --------
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)

# -------- Confusion Matrix --------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# -------- Plot CM --------
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("/Users/nahidkhan/Local Drive/Research/Landmark_Data/confusion_matrix.png", dpi=200)
plt.show()

print("\nSaved confusion_matrix.png in Landmark_Data/")
