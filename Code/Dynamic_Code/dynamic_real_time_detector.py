# dynamic_real_time_detector.py

import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = "models"
ORIGINAL_DYNAMIC_PATH = "Dynamic_Data"
CONFIDENCE_THRESHOLD = 0.8

# --- Helper Function for Dynamic Signs ---
def extract_landmarks_dynamic(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- Main Program ---
if __name__ == "__main__":
    # --- Load the Dynamic Model ---
    try:
        dynamic_model = load_model(os.path.join(MODEL_PATH, 'dynamic_model.h5'))
        DYNAMIC_ACTIONS = np.array([d for d in os.listdir(ORIGINAL_DYNAMIC_PATH) if os.path.isdir(os.path.join(ORIGINAL_DYNAMIC_PATH, d))])
        dynamic_sequence_length = dynamic_model.input_shape[1]
        print(f"Successfully loaded Dynamic Model. Actions: {DYNAMIC_ACTIONS}")
    except Exception as e:
        print(f"\nError: Could not load dynamic_model.h5. Reason: {e}")
        print("Please ensure the model is in the 'models' directory. Exiting.")
        exit()

    # --- Initialize Real-time Variables ---
    sequence = []
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
            
            # --- Dynamic Prediction Logic ---
            dynamic_keypoints = extract_landmarks_dynamic(results)
            sequence.append(dynamic_keypoints)
            sequence = sequence[-dynamic_sequence_length:]

            if len(sequence) == dynamic_sequence_length:
                dynamic_res = dynamic_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                if np.max(dynamic_res) > CONFIDENCE_THRESHOLD:
                    prediction_text = DYNAMIC_ACTIONS[np.argmax(dynamic_res)]
                else:
                    prediction_text = "..."
            else:
                prediction_text = "..."

            # Display the result
            (text_width, text_height), baseline = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(image, (0, 0), (text_width + 20, text_height + 20), (0, 0, 0), -1)
            cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Dynamic Sign Language Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()