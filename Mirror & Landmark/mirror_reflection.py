import cv2
import os

def create_mirror_images_for_class(input_class_folder, output_class_folder):
    """
    Creates mirror reflections for all images in a specific class folder
    and saves them to the output class folder.

    Args:
        input_class_folder (str): Path to the class folder with original images (e.g., '01_Raw_Data/L1').
        output_class_folder (str): Path to save the mirrored images (e.g., '02_Augmented_Data/Mirrored_Images/L1').
    """

    # Create the output class folder if it doesn't exist
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)
        print(f"  Creating output folder: {output_class_folder}")

    captured_count = 0 # To count processed images

    # Loop through files in the input class folder
    for filename in os.listdir(input_class_folder):
        # Ensure it's an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(input_class_folder, filename)

            # Load the image
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

            # Save the mirrored image
            cv2.imwrite(output_file_path, mirrored_img)
            captured_count += 1
            # print(f"  Saved: {output_file_path}") # You can uncomment this line to see every file save

    print(f"Created {captured_count} mirror images for '{os.path.basename(input_class_folder)}'.")


# --- Main execution ---
if __name__ == "__main__":
    
    # Path to the main folder containing original class folders (e.g., L1, L2, L3...)
    # This should match the 'DATA_PATH' from your static_data_collector.py
    base_raw_data_folder = '/Users/nahidkhan/Local Drive/Research/Sample' # Update this path if needed

    # Path to the main folder where mirrored images will be saved
    base_augmented_data_folder = '/Users/nahidkhan/Local Drive/Research/Augmented_Data/Mirrored_Images'

    print("Starting mirror reflection process...")

    # Ensure the base output folder exists
    if not os.path.exists(base_augmented_data_folder):
        os.makedirs(base_augmented_data_folder)
        print(f"Base output folder created: {base_augmented_data_folder}")

    # Loop through each class folder (e.g., L1, L2, L3...)
    # os.listdir() gets all file/folder names inside 'base_raw_data_folder'
    for class_folder_name in os.listdir(base_raw_data_folder):
        input_class_folder_path = os.path.join(base_raw_data_folder, class_folder_name)

        # Ensure it's a directory and not a hidden file/folder
        if os.path.isdir(input_class_folder_path) and not class_folder_name.startswith('.'):
            print(f"\nProcessing folder: '{class_folder_name}'")

            # Create the path for the output folder
            output_class_folder_path = os.path.join(base_augmented_data_folder, class_folder_name)

            # Call the function to create mirror images for this class
            create_mirror_images_for_class(input_class_folder_path, output_class_folder_path)

    print("\nMirror reflection process completed for all classes.")