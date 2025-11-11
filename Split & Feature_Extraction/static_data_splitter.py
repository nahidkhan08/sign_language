# static_data_splitter.py (Revised for Raw + Mirrored)
import os
import shutil
import random
from collections import defaultdict

# --- Configuration ---
# Define the source directories
RAW_IMAGE_DIR = os.path.join('/Users/nahidkhan/Local Drive/Research/Dataset', 'Static')
MIRRORED_IMAGE_DIR = os.path.join('/Users/nahidkhan/Local Drive/Research/Dataset', 'Mirrored_Images')

# Define the main output directory for split IMAGES (not features yet)
SPLIT_IMAGE_DIR = '/Users/nahidkhan/Local Drive/Research/Dataset/Split_Image' # Intermediate folder for split images

# Define the split ratio
SPLIT_RATIO = (0.8, 0.1, 0.1) # Train, Validation, Test

def split_data(raw_dir, mirrored_dir, output_base_dir, split_ratio):
    """
    Splits data from raw and mirrored directories into train, val, and test sets.
    """
    print(f"Output directory: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'test'), exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"Error: Raw image directory not found: {raw_dir}")
        return

    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith('.')]
    print(f"Found classes: {classes}")

    for class_name in classes:
        print(f"\nProcessing class: {class_name}")

        # --- Collect all files for the class ---
        all_class_files = []
        # Add raw images
        class_raw_path = os.path.join(raw_dir, class_name)
        if os.path.exists(class_raw_path):
            all_class_files.extend([(os.path.join(class_raw_path, f), f) for f in os.listdir(class_raw_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Add mirrored images
        class_mirrored_path = os.path.join(mirrored_dir, class_name)
        if os.path.exists(class_mirrored_path):
             all_class_files.extend([(os.path.join(class_mirrored_path, f), f) for f in os.listdir(class_mirrored_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
             print(f"  Warning: Mirrored directory not found for class {class_name}, skipping mirrored images.")


        if not all_class_files:
            print(f"  No image files found for class {class_name}. Skipping.")
            continue

        # Shuffle the combined list
        random.shuffle(all_class_files)

        # --- Calculate split points ---
        total_files = len(all_class_files)
        train_split_point = int(total_files * split_ratio[0])
        val_split_point = int(total_files * (split_ratio[0] + split_ratio[1]))

        train_files = all_class_files[:train_split_point]
        val_files = all_class_files[train_split_point:val_split_point]
        test_files = all_class_files[val_split_point:]

        # --- Create destination directories ---
        train_dest = os.path.join(output_base_dir, 'train', class_name)
        val_dest = os.path.join(output_base_dir, 'val', class_name)
        test_dest = os.path.join(output_base_dir, 'test', class_name)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        # --- Copy files ---
        def copy_files(files, destination_dir):
            copied_count = 0
            for source_path, file_name in files:
                try:
                    shutil.copyfile(source_path, os.path.join(destination_dir, file_name))
                    copied_count += 1
                except Exception as e:
                    print(f"  Error copying {source_path}: {e}")
            return copied_count

        train_count = copy_files(train_files, train_dest)
        val_count = copy_files(val_files, val_dest)
        test_count = copy_files(test_files, test_dest)

        print(f"  Split complete: {train_count} train, {val_count} val, {test_count} test images.")

if __name__ == "__main__":
    print("Starting data splitting process (Raw + Mirrored)...")
    split_data(RAW_IMAGE_DIR, MIRRORED_IMAGE_DIR, SPLIT_IMAGE_DIR, SPLIT_RATIO)
    print("\nData splitting process completed successfully!")
    print(f"Split images are available in the '{SPLIT_IMAGE_DIR}' directory.")