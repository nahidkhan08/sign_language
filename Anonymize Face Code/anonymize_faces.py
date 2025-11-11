import cv2
import numpy as np
import os

# --- Configuration ---
PROTOTXT_PATH = "Mirror & Landmark/deploy.prototxt.txt"# Path to the .prototxt file
MODEL_PATH = "Mirror & Landmark/res10_300x300_ssd_iter_140000.caffemodel" # Path to the .caffemodel file
CONFIDENCE_THRESHOLD = 0.5 # Minimum probability to accept a detection as a face
BLUR_KERNEL_SIZE = (99, 99) # How much blur to apply (odd numbers, larger means more blur)

# Folders to process (relative to where the script is run)
IMAGE_FOLDERS = [
    os.path.join('/Users/nahidkhan/Local Drive/Research/Dataset', 'Static'),
    os.path.join('/Users/nahidkhan/Local Drive/Research/Dataset', 'Mirrored_Images')
]

# --- Load the Face Detector Model ---
try:
    print("[INFO] Loading face detector model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except cv2.error as e:
    print(f"[ERROR] Could not load model. Ensure '{PROTOTXT_PATH}' and '{MODEL_PATH}' are in the correct directory.")
    print(f"OpenCV Error: {e}")
    exit()

# --- Function to Anonymize Faces ---
def anonymize_face_pixelated(image, net, confidence_threshold, kernel_size):
    """Detects faces and applies Gaussian blur."""
    (h, w) = image.shape[:2]
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    anonymized = False # Flag to check if any face was blurred

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            anonymized = True
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box stays within image bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Extract the face ROI
            face = image[startY:endY, startX:endX]

            # Apply Gaussian blur
            try:
                # Adjust kernel size if face ROI is too small
                k_w = min(kernel_size[0], face.shape[1] - (face.shape[1] % 2 == 0))
                k_h = min(kernel_size[1], face.shape[0] - (face.shape[0] % 2 == 0))
                if k_w > 0 and k_h > 0:
                    blurred_face = cv2.GaussianBlur(face, (k_w, k_h), 0)
                    # Put the blurred face back into the original image
                    image[startY:endY, startX:endX] = blurred_face
                else:
                    print(f"    [Warning] Face ROI too small for blurring: {(startX, startY, endX, endY)}")
            except Exception as e:
                 print(f"    [Error] Could not blur face ROI: {e}")


    return image, anonymized

# --- Main Processing Loop ---
if __name__ == "__main__":
    print("\nStarting face anonymization process...")

    total_processed = 0
    total_anonymized = 0

    for base_folder in IMAGE_FOLDERS:
        if not os.path.exists(base_folder):
            print(f"[Warning] Folder not found, skipping: {base_folder}")
            continue

        print(f"\nProcessing folder: {base_folder}")
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(base_folder):
            # Skip hidden directories like .DS_Store
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue # Skip empty folders or folders without images

            print(f"  Processing directory: {root} ({len(image_files)} images)")

            for filename in image_files:
                image_path = os.path.join(root, filename)
                try:
                    # Read the image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"    [Warning] Could not read image: {filename}. Skipping.")
                        continue

                    total_processed += 1
                    # Anonymize faces
                    anonymized_image, was_anonymized = anonymize_face_pixelated(
                        image, net, CONFIDENCE_THRESHOLD, BLUR_KERNEL_SIZE
                    )

                    # --- IMPORTANT: Overwrite the original image ---
                    if was_anonymized:
                        cv2.imwrite(image_path, anonymized_image)
                        total_anonymized += 1
                        # print(f"    Anonymized and saved: {filename}")
                    # else:
                        # print(f"    No faces detected above threshold in: {filename}")

                    # --- Alternative: Save to a new directory (uncomment to use) ---
                    # output_dir = root.replace(base_folder, base_folder + "_anonymized", 1)
                    # os.makedirs(output_dir, exist_ok=True)
                    # output_path = os.path.join(output_dir, filename)
                    # cv2.imwrite(output_path, anonymized_image)
                    # print(f"    Saved anonymized image to: {output_path}")

                except Exception as e:
                    print(f"    [Error] Failed processing {filename}: {e}")

    print("\n--------------------")
    print("Anonymization Summary:")
    print(f"Total images processed: {total_processed}")
    print(f"Total images where faces were detected and blurred: {total_anonymized}")
    print("--------------------")
    print("Process completed.")