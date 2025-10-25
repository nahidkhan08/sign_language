# final_stable_trainer_static.py

# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt

# --- (The Configuration, Augmentation, and Data Loading functions remain exactly the same) ---
# ...
# ... (Paste your previous Config, augment_flip, augment_noise, and load_and_prepare_data functions here) ...
# ...

# STATIC-ONLY PATHS
PROCESSED_DATA_PATH = '/Users/nahidkhan/Local Drive/Research/New_Model_Train/Feature_Extraction'
ORIGINAL_STATIC_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset/Static'
MODEL_SAVE_PATH = '/Users/nahidkhan/Local Drive/Research/New_Model_Train/new model'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def augment_flip(landmarks):
    # ... (code from previous version) ...
    flipped_landmarks = landmarks.copy()
    num_coords = len(flipped_landmarks)
    if num_coords == 126:
        for i in range(0, num_coords, 3): flipped_landmarks[i] = 1.0 - flipped_landmarks[i]
        lh_data, rh_data = flipped_landmarks[:63].copy(), flipped_landmarks[63:].copy()
        flipped_landmarks[:63], flipped_landmarks[63:] = rh_data, lh_data
    elif num_coords == 258:
        pose_coords = 132
        for i in range(0, pose_coords, 4): flipped_landmarks[i] = 1.0 - flipped_landmarks[i]
        for i in range(pose_coords, num_coords, 3): flipped_landmarks[i] = 1.0 - flipped_landmarks[i]
        lh_data = flipped_landmarks[pose_coords:pose_coords+63].copy()
        rh_data = flipped_landmarks[pose_coords+63:].copy()
        flipped_landmarks[pose_coords:pose_coords+63] = rh_data
        flipped_landmarks[pose_coords+63:] = lh_data
    return flipped_landmarks

def augment_noise(landmarks, scale=0.003):
    return landmarks + np.random.normal(0, scale, landmarks.shape)

def load_and_prepare_data(data_path, actions):
    # ... (code from previous version) ...
    label_map = {label: num for num, label in enumerate(actions)}
    X, y = {'train': [], 'val': [], 'test': []}, {'train': [], 'val': [], 'test': []}
    for action in actions:
        for split in ['train', 'val', 'test']:
            action_path = os.path.join(data_path, split, action)
            if not os.path.exists(action_path): continue
            for file_name in os.listdir(action_path):
                if file_name.endswith('.npy'):
                    res = np.load(os.path.join(action_path, file_name))
                    X[split].append(res)
                    y[split].append(label_map[action])
                    if split == 'train':
                        X['train'].append(augment_flip(res))
                        y['train'].append(label_map[action])
                        X['train'].append(augment_noise(res))
                        y['train'].append(label_map[action])
    return X, y

# --- Main Program (STATIC only) ---
if __name__ == "__main__":
    model_type = 'static'

    # ... (Static model part remains the same) ...
    actions = np.array([d for d in os.listdir(ORIGINAL_STATIC_PATH) if os.path.isdir(os.path.join(ORIGINAL_STATIC_PATH, d))])
    X_dict, y_dict = load_and_prepare_data(PROCESSED_DATA_PATH, actions)
    X_train, y_train = np.array(X_dict['train']), to_categorical(y_dict['train']).astype(int)
    X_val, y_val = np.array(X_dict['val']), to_categorical(y_dict['val']).astype(int)
    X_test, y_test = np.array(X_dict['test']), to_categorical(y_dict['test']).astype(int)
    model = Sequential([Dense(128, activation='relu', input_shape=(X_train.shape[1],)),Dense(64, activation='relu'),Dense(32, activation='relu'),Dense(actions.shape[0], activation='softmax')])
    model_filename = 'static_model_final.h5'
    callbacks = [EarlyStopping(monitor='val_categorical_accuracy', patience=15, verbose=1, restore_best_weights=True)]

    # --- MODIFICATION 2: Add 'clipnorm' to the Adam optimizer for Gradient Clipping ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    print(f"\n--- Starting Augmented {model_type.upper()} Model Training ---")
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks)

    # --- Accuracy plot (added) ---
    try:
        plt.figure(figsize=(8,5))
        plt.plot(history.history.get('categorical_accuracy', []), label='Train Accuracy')
        plt.plot(history.history.get('val_categorical_accuracy', []), label='Validation Accuracy')
        plt.title(f'{model_type.upper()} Model Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_PATH, f'{model_type}_accuracy.png'), dpi=200)
        plt.show()
    except Exception as e:
        print("Accuracy plot failed:", e)

    # --- (Evaluation and saving code remains the same) ---
    print("\n--- Model Evaluation on Test Set (using best weights) ---")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=actions, zero_division=0))
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title(f'Confusion Matrix for {model_type.upper()} Model')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    model.save(os.path.join(MODEL_SAVE_PATH, model_filename))
    print(f"Training complete. Model saved as '{model_filename}'")
