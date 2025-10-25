import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic and Drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_image(image, holistic_model):
    """
    Process a single image, draw all holistic landmarks on it,
    and return the annotated image.
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find landmarks
    results = holistic_model.process(image_rgb)

    # Create a copy of the original image to draw on
    annotated_image = image.copy()
    
    # --- Draw all landmarks on the image copy ---

    # 1. Draw Face landmarks (contours only)
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None, # Do not draw individual points
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )

    # 2. Draw Pose landmarks
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # 3. Draw Left Hand landmarks
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
    )

    # 4. Draw Right Hand landmarks
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
    )
    
    return annotated_image

def process_and_annotate_dataset(base_data_folders, output_folder):
    """
    Loops through all class folders, reads each image (raw and augmented),
    draws landmarks on it, and saves the annotated image.
    """
    
    try:
        class_folders = [f for f in os.listdir(base_data_folders[0]) if os.path.isdir(os.path.join(base_data_folders[0], f)) and not f.startswith('.')]
    except FileNotFoundError:
        print(f"Error: Base folder not found: {base_data_folders[0]}")
        print("Please ensure '01_Raw_Data' folder exists.")
        return

    print(f"Found {len(class_folders)} classes. Starting annotation...")

    # Create the main output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Initialize Holistic model once
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
        
        for class_name in class_folders:
            print(f"\nProcessing class: {class_name}")
            
            # Create the class-specific output folder (e.g., .../Annotated_Images/L1)
            output_class_folder = os.path.join(output_folder, class_name)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

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
                        
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"    Warning: Could not read {image_path}")
                            continue
                        
                        # Get the image with landmarks drawn on it
                        annotated_img = draw_landmarks_on_image(image, holistic)
                        
                        # Define the output path and save the annotated image
                        output_file_path = os.path.join(output_class_folder, image_name)
                        cv2.imwrite(output_file_path, annotated_img)

            print(f"  Finished annotating images for '{class_name}'.")

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. List of folders to read images from
    DATA_FOLDERS = [
        '/Users/nahidkhan/Local Drive/Research/Sample',
        '/Users/nahidkhan/Local Drive/Research/Augmented_Data/Mirrored_Images'
    ]
    
    # 2. Folder to save the final annotated images
    OUTPUT_ANNOTATED_FOLDER = '/Users/nahidkhan/Local Drive/Research/Landmark Image'

    process_and_annotate_dataset(DATA_FOLDERS, OUTPUT_ANNOTATED_FOLDER)

    print("\nLandmark visualization process completed.")