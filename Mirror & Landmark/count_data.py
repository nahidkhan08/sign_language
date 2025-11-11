import os
import collections

# --- Configuration ---
# Define the base paths according to your folder structure
BASE_IMAGE_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset'
RAW_IMAGE_DIR = os.path.join(BASE_IMAGE_PATH, 'Static')
MIRRORED_IMAGE_DIR = os.path.join(BASE_IMAGE_PATH, 'Mirrored_Images')
PROCESSED_FEATURES_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset/02_Processed_Features_NPY'
OUTPUT_FILENAME = '/Users/nahidkhan/Local Drive/Research/Dataset/dataset_stats.txt' # Name of the output text file

# Expected image and feature file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
FEATURE_EXTENSION = '.npy'

# --- Function to count files in a directory ---
def count_files(directory, extensions):
    """Counts files with specific extensions in a directory."""
    count = 0
    if os.path.exists(directory):
        # Use listdir for immediate subdirectories only, more efficient than walk
        try:
            for item in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, item)) and item.lower().endswith(extensions):
                    count += 1
        except FileNotFoundError:
            # Handle cases where the class directory might be missing in some splits/types
            pass
        except Exception as e:
            print(f"  Warning: Could not read directory {directory}: {e}")
    return count

# --- Main Logic ---
if __name__ == "__main__":
    class_stats = collections.defaultdict(lambda: {
        'raw': 0, 'mirrored': 0, 'train': 0, 'val': 0, 'test': 0, 'total_features': 0
    })
    total_stats = collections.defaultdict(int)

    # --- Get class labels (assuming Raw_Images has all base classes) ---
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"Error: Raw image directory not found at '{RAW_IMAGE_DIR}'")
        exit()

    try:
        class_labels = sorted([
            d for d in os.listdir(RAW_IMAGE_DIR)
            if os.path.isdir(os.path.join(RAW_IMAGE_DIR, d)) and not d.startswith('.')
        ])
    except Exception as e:
        print(f"Error reading class directories from '{RAW_IMAGE_DIR}': {e}")
        exit()


    if not class_labels:
        print(f"Error: No class subdirectories found in '{RAW_IMAGE_DIR}'")
        exit()

    print(f"Found {len(class_labels)} classes. Counting files...")

    # --- Count files for each class ---
    for class_label in class_labels:
        # Count Raw Images
        raw_count = count_files(os.path.join(RAW_IMAGE_DIR, class_label), IMAGE_EXTENSIONS)
        class_stats[class_label]['raw'] = raw_count
        total_stats['raw'] += raw_count

        # Count Mirrored Images
        mirrored_count = count_files(os.path.join(MIRRORED_IMAGE_DIR, class_label), IMAGE_EXTENSIONS)
        class_stats[class_label]['mirrored'] = mirrored_count
        total_stats['mirrored'] += mirrored_count

        # Count Processed Features (Train, Val, Test)
        train_count = count_files(os.path.join(PROCESSED_FEATURES_PATH, 'train', class_label), FEATURE_EXTENSION)
        val_count = count_files(os.path.join(PROCESSED_FEATURES_PATH, 'val', class_label), FEATURE_EXTENSION)
        test_count = count_files(os.path.join(PROCESSED_FEATURES_PATH, 'test', class_label), FEATURE_EXTENSION)

        class_stats[class_label]['train'] = train_count
        class_stats[class_label]['val'] = val_count
        class_stats[class_label]['test'] = test_count
        class_stats[class_label]['total_features'] = train_count + val_count + test_count

        total_stats['train'] += train_count
        total_stats['val'] += val_count
        total_stats['test'] += test_count
        total_stats['total_features'] += class_stats[class_label]['total_features']

    # --- Write Results to Text File ---
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write("### Dataset Statistics\n\n")
            f.write(f"This dataset contains a total of **{total_stats['total_features']}** processed feature samples across **{len(class_labels)}** static Bangla Sign Language classes.\n")
            f.write("The data is organized into raw images, mirrored images, and processed NumPy feature files (126 features per sample).\n\n")
            f.write("**Class Distribution:**\n\n")

            # Write Header
            header = "| Class Label | Raw Images | Mirrored Images | Total Features | Train Features | Val Features | Test Features |\n"
            separator = "|-------------|------------|-----------------|----------------|----------------|--------------|---------------|\n"
            f.write(header)
            f.write(separator)

            # Write Data Rows
            for class_label in class_labels:
                stats = class_stats[class_label]
                f.write(f"| {class_label:<11} | {stats['raw']:<10} | {stats['mirrored']:<15} | {stats['total_features']:<14} | {stats['train']:<14} | {stats['val']:<12} | {stats['test']:<13} |\n")

            # Write Footer (Totals)
            f.write(separator)
            f.write(f"| **Total** | **{total_stats['raw']:<8}** | **{total_stats['mirrored']:<13}** | **{total_stats['total_features']:<12}** | **{total_stats['train']:<12}** | **{total_stats['val']:<10}** | **{total_stats['test']:<11}** |\n")

        print(f"\nCounting complete. Statistics saved to '{OUTPUT_FILENAME}'.")

    except IOError as e:
        print(f"Error writing to file '{OUTPUT_FILENAME}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")