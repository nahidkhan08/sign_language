# static_real_time_detector.py

import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = '/Users/nahidkhan/Local Drive/Research/New_Model_Train/new model'
ORIGINAL_STATIC_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset/Static'
CONFIDENCE_THRESHOLD = 0.8

# --- Helper Function for Static Signs ---
def extract_landmarks_static(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# --- Main Program ---
if __name__ == "__main__":
    # --- Load the Static Model ---
    try:
        static_model = load_model(os.path.join(MODEL_PATH, 'static_model_final.h5'))
        STATIC_ACTIONS = np.array([d for d in os.listdir(ORIGINAL_STATIC_PATH) if os.path.isdir(os.path.join(ORIGINAL_STATIC_PATH, d))])
        print(f"Successfully loaded Static Model. Actions: {STATIC_ACTIONS}")
    except Exception as e:
        print(f"\nError: Could not load static_model.h5. Reason: {e}")
        print("Please ensure the model is in the 'models' directory. Exiting.")
        exit()

    # --- Initialize Real-time Variables ---
    prediction_text = "..."

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # --- Start Real-time Detection Loop ---
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # --- Static Prediction Logic ---
            static_keypoints = extract_landmarks_static(results)
            static_res = static_model.predict(np.expand_dims(static_keypoints, axis=0), verbose=0)[0]

            if np.max(static_res) > CONFIDENCE_THRESHOLD:
                prediction_text = STATIC_ACTIONS[np.argmax(static_res)]
            else:
                prediction_text = "..."

            # Display the result
            (text_width, text_height), baseline = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(image, (0, 0), (text_width + 20, text_height + 20), (0, 0, 0), -1)
            cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Static Sign Language Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()