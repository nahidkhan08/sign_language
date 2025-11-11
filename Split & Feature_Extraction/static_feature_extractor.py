# static_feature_extractor.py
import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic

# --- Configuration ---
# Input directory: Where the split IMAGE files are
SPLIT_IMAGE_DATA_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset/Split_Image'

# Output directory: Where the split FEATURE (.npy) files will be saved
OUTPUT_FEATURE_DATA_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset/02_Processed_Features_NPY'

# --- Landmark Extraction Function for STATIC data ---
def extract_landmarks_static(results):
    """
    Extracts landmarks ONLY for left and right hands (126 features total).
    Fills with zeros if a hand is not detected.
    """
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh]) # Total 63 + 63 = 126 features

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting STATIC feature extraction process...")

    os.makedirs(OUTPUT_FEATURE_DATA_PATH, exist_ok=True)

    # Use MediaPipe Holistic
    # static_image_mode=True is better for processing individual image files
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:

        # Loop through train, val, test splits in the image directory
        for split_part in ['train', 'val', 'test']:
            split_image_path = os.path.join(SPLIT_IMAGE_DATA_PATH, split_part)
            output_feature_split_path = os.path.join(OUTPUT_FEATURE_DATA_PATH, split_part)
            os.makedirs(output_feature_split_path, exist_ok=True)

            if not os.path.exists(split_image_path):
                print(f"Warning: Split image directory '{split_image_path}' not found. Skipping.")
                continue

            print(f"\n--- Processing '{split_part}' image set ---")

            # Loop through each class in the split
            for class_name in os.listdir(split_image_path):
                class_image_path = os.path.join(split_image_path, class_name)
                if not os.path.isdir(class_image_path):
                    continue

                print(f"  Processing class: '{class_name}'")

                output_feature_class_path = os.path.join(output_feature_split_path, class_name)
                os.makedirs(output_feature_class_path, exist_ok=True)

                # Process each image file in the class folder
                processed_count = 0
                for file_name in os.listdir(class_image_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_file_path = os.path.join(class_image_path, file_name)

                        # Read image
                        frame = cv2.imread(image_file_path)
                        if frame is None:
                            print(f"    Warning: Could not read image {image_file_path}. Skipping.")
                            continue

                        # Convert BGR to RGB for MediaPipe
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Process image to get landmarks
                        results = holistic.process(image_rgb)

                        # Extract only hand landmarks (126 features)
                        keypoints = extract_landmarks_static(results)

                        # Define path to save .npy file
                        npy_file_name = os.path.splitext(file_name)[0] + '.npy'
                        npy_path = os.path.join(output_feature_class_path, npy_file_name)

                        # Save the features as .npy file
                        np.save(npy_path, keypoints)
                        processed_count += 1

                print(f"    Extracted features for {processed_count} images.")

    print("\nStatic feature extraction completed successfully!")
    print(f"Feature files (.npy) are available in '{OUTPUT_FEATURE_DATA_PATH}'.")