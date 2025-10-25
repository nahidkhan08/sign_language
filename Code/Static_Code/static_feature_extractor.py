# static_feature_extractor.py

import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic

# --- Configuration ---
SPLIT_DATA_PATH = '/Users/nahidkhan/Local Drive/Research/New_Model_Train/Split'
OUTPUT_DATA_PATH = '/Users/nahidkhan/Local Drive/Research/New_Model_Train/Feature_Extraction'

# --- Landmark Extraction Function for STATIC data ---
def extract_landmarks_static(results):
    """
    Extracts landmarks for ONLY left and right hands.
    """
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting the STATIC feature extraction process...")

    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for split_part in ['train', 'val', 'test']:
            split_path = os.path.join(SPLIT_DATA_PATH, split_part)
            output_split_path = os.path.join(OUTPUT_DATA_PATH, split_part)
            os.makedirs(output_split_path, exist_ok=True)
            
            if not os.path.exists(split_path):
                print(f"Warning: Directory '{split_path}' not found. Skipping.")
                continue

            print(f"\n--- Processing '{split_part}' set for STATIC data ---")
            
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                print(f"Processing static class: '{class_name}'")
                
                output_class_path = os.path.join(output_split_path, class_name)
                os.makedirs(output_class_path, exist_ok=True)

                # Process only image files
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.jpg'):
                        file_path = os.path.join(class_path, file_name)
                        frame = cv2.imread(file_path)
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image)
                        keypoints = extract_landmarks_static(results)
                        npy_path = os.path.join(output_class_path, file_name.split('.')[0])
                        np.save(npy_path, keypoints)

    print("\nStatic feature extraction completed successfully!")