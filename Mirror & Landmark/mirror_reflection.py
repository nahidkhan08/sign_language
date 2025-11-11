import cv2
import os

def create_mirror_images_for_class(input_class_folder, output_class_folder):
    """
    Creates mirror reflections for all images in a specific class folder
    and saves them to the output class folder.

    Args:
        input_class_folder (str): Path to the class folder with original images.
        output_class_folder (str): Path where the mirrored images will be saved.
    """
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)
        print(f"  Creating output folder: {output_class_folder}")

    captured_count = 0
    for filename in os.listdir(input_class_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(input_class_folder, filename)
            img = cv2.imread(file_path)
            
            if img is None:
                print(f"  Warning: Could not load image: {file_path}")
                continue

            # Create mirror reflection (horizontal flip)
            mirrored_img = cv2.flip(img, 1)

            # Create new filename (e.g., L1_1.jpg -> L1_1_mirror.jpg)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_mirror{ext}"
            output_file_path = os.path.join(output_class_folder, output_filename)

            cv2.imwrite(output_file_path, mirrored_img)
            captured_count += 1

    print(f"Created {captured_count} mirror images for '{os.path.basename(input_class_folder)}'.")

if __name__ == "__main__":
    # --- Define your paths here ---
    base_raw_data_folder = os.path.join('/Users/nahidkhan/Local Drive/Research/Dataset', 'Static')
    base_augmented_data_folder = os.path.join('/Users/nahidkhan/Local Drive/Research/Dataset', 'Mirrored_Images')
    # --------------------------------

    print("Starting mirror reflection (data augmentation) process...")
    
    # !! This is the important check from SCRIPT 1 !!
    if not os.path.exists(base_raw_data_folder):
        print(f"Error: Base raw data folder not found: {base_raw_data_folder}")
        exit()

    if not os.path.exists(base_augmented_data_folder):
        os.makedirs(base_augmented_data_folder)
        print(f"Base output folder created: {base_augmented_data_folder}")

    for class_folder_name in os.listdir(base_raw_data_folder):
        input_class_folder_path = os.path.join(base_raw_data_folder, class_folder_name)
        
        # Ensure it's a directory and not a hidden file (like .DS_Store on macOS)
        if os.path.isdir(input_class_folder_path) and not class_folder_name.startswith('.'):
            print(f"\nProcessing folder: '{class_folder_name}'")
            
            output_class_folder_path = os.path.join(base_augmented_data_folder, class_folder_name)
            
            create_mirror_images_for_class(input_class_folder_path, output_class_folder_path)

    print("\nMirror reflection process completed for all classes.")