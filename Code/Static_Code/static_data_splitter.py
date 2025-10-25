# static_data_splitter.py

# Import necessary libraries for file operations and randomization
import os
import shutil
import random

# --- Configuration ---
# Define the source directory where your collected data is stored
STATIC_SOURCE_DIR = "/Users/nahidkhan/Local Drive/Research/Dataset/Static"

# Define the main output directory where the split data will be saved
BASE_OUTPUT_DIR = '/Users/nahidkhan/Local Drive/Research/New_Model_Train/Split'

# Define the split ratio for training, validation, and testing sets
# Example: 80% for training, 10% for validation, 10% for testing
SPLIT_RATIO = (0.8, 0.1, 0.1)


def split_class_data(class_source_path, class_name, base_output_dir):
    """
    This function takes a folder of data for a single class, shuffles it,
    and splits it into train, val, and test sets.
    """
    # Create the train, val, and test directories if they don't exist
    train_dir = os.path.join(base_output_dir, 'train', class_name)
    val_dir = os.path.join(base_output_dir, 'val', class_name)
    test_dir = os.path.join(base_output_dir, 'test', class_name)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get a list of all files in the source class folder
    all_files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
    
    # Shuffle the files randomly to ensure unbiased distribution
    random.shuffle(all_files)
    
    # Calculate the split points
    total_files = len(all_files)
    train_split_point = int(total_files * SPLIT_RATIO[0])
    val_split_point = int(total_files * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
    
    # Slice the file list into three sets
    train_files = all_files[:train_split_point]
    val_files = all_files[train_split_point:val_split_point]
    test_files = all_files[val_split_point:]
    
    # Function to copy files to their new destination
    def copy_files(files, destination_dir):
        for file_name in files:
            source_file = os.path.join(class_source_path, file_name)
            destination_file = os.path.join(destination_dir, file_name)
            shutil.copyfile(source_file, destination_file)
            
    # Copy the files into the respective directories
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    print(f"Split class '{class_name}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting the static data splitting process...")
    
    # --- Process all directories inside the STATIC source folder ---
    if os.path.exists(STATIC_SOURCE_DIR):
        print(f"\nProcessing static data from '{STATIC_SOURCE_DIR}'...")
        for class_name in os.listdir(STATIC_SOURCE_DIR):
            class_path = os.path.join(STATIC_SOURCE_DIR, class_name)
            if os.path.isdir(class_path):
                split_class_data(class_path, class_name, BASE_OUTPUT_DIR)
    else:
        print(f"Warning: Directory '{STATIC_SOURCE_DIR}' not found. Skipping.")
        
    print("\nStatic data splitting process completed successfully!")
    print(f"Split data is available in the '{BASE_OUTPUT_DIR}' directory.")