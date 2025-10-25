import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image, holistic_model):
    """
    Process a single image and extract Pose, Face, Left Hand, and Right Hand landmarks.
    Returns a single flat numpy array of all landmark coordinates.
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find landmarks
    results = holistic_model.process(image_rgb)

    # --- Extract landmarks and flatten them ---
    # We create a flat list. If landmarks are not detected, we fill with zeros.

    # 1. Pose (33 landmarks * 4 coords = 132 features)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4) # 33 landmarks, 4 coordinates (x, y, z, visibility)

    # 2. Face (468 landmarks * 3 coords = 1404 features)
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3) # 468 landmarks, 3 coordinates (x, y, z)

    # 3. Left Hand (21 landmarks * 3 coords = 63 features)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3) # 21 landmarks, 3 coordinates (x, y, z)

    # 4. Right Hand (21 landmarks * 3 coords = 63 features)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3) # 21 landmarks, 3 coordinates (x, y, z)

    # Concatenate all features into a single row
    # Total features = 132 (pose) + 1404 (face) + 63 (left) + 63 (right) = 1662 features
    return np.concatenate([pose, face, left_hand, right_hand])

def process_dataset(base_data_folders, output_folder):
    """
    Loops through all class folders in the base_data_folders,
    extracts landmarks from each image (raw and augmented),
    and saves the results as one .npy file per class.
    """
    
    # Get the list of classes (e.g., L1, L2) from the Raw_Data folder
    try:
        class_folders = [f for f in os.listdir(base_data_folders[0]) if os.path.isdir(os.path.join(base_data_folders[0], f)) and not f.startswith('.')]
    except FileNotFoundError:
        print(f"Error: Base folder not found: {base_data_folders[0]}")
        print("Please ensure '01_Raw_Data' folder exists and is correctly named.")
        return

    print(f"Found {len(class_folders)} classes: {class_folders}")

    # Create the main output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Initialize Holistic model once
    # static_image_mode=True is important for processing individual images
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
        
        for class_name in class_folders:
            print(f"\nProcessing class: {class_name}")
            
            class_landmarks_list = [] # To store landmarks for all images in this class
            
            # Loop through each base folder (Raw and Augmented)
            for base_folder in base_data_folders:
                class_folder_path = os.path.join(base_folder, class_name)
                
                if not os.path.exists(class_folder_path):
                    print(f"  Skipping: Folder not found {class_folder_path}")
                    continue
                    
                print(f"  Reading from: {class_folder_path}")
                
                # Loop through each image in the class folder
                for image_name in os.listdir(class_folder_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_folder_path, image_name)
                        
                        # Read the image
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"    Warning: Could not read {image_path}")
                            continue
                        
                        # Extract landmarks
                        landmarks = extract_landmarks(image, holistic)
                        
                        # Add the flat array of landmarks to our list
                        class_landmarks_list.append(landmarks)

            # After processing all images for this class, convert list to NumPy array
            if class_landmarks_list:
                class_landmarks_array = np.array(class_landmarks_list)
                
                # Save the array to a .npy file
                output_file_path = os.path.join(output_folder, f"{class_name}.npy")
                np.save(output_file_path, class_landmarks_array)
                
                print(f"  Saved landmarks for '{class_name}':")
                print(f"  - Shape: {class_landmarks_array.shape}") # (num_images, 1662)
                print(f"  - File: {output_file_path}")
            else:
                print(f"  No images found or processed for class '{class_name}'.")

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. List of folders to read images from
    # We process both original and mirrored images
    DATA_FOLDERS = [
        '01_Raw_Data',
        '02_Augmented_Data/Mirrored_Images'
    ]
    
    # 2. Folder to save the final .npy files
    OUTPUT_LANDMARKS_FOLDER = '03_Processed_Features/Landmarks_Holistic_Numpy'

    # Install dependencies if needed
    # pip install mediapipe numpy opencv-python

    process_dataset(DATA_FOLDERS, OUTPUT_LANDMARKS_FOLDER)

    print("\nLandmark extraction process completed.")