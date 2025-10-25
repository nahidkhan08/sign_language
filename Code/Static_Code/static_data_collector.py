# static_data_collector.py

# Import necessary libraries
import cv2
import os

# --- Logic for STATIC data collection (Image Capture) ---
DATA_PATH = '/Users/nahidkhan/Local Drive/Research/Dataset/Static'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
            
sign_name = input("Enter the static sign name (e.g., 'o', 'ko', '1'): ")
sign_path = os.path.join(DATA_PATH, sign_name)
if not os.path.exists(sign_path):
    os.makedirs(sign_path)

cap = cv2.VideoCapture(0)
count = len(os.listdir(sign_path)) + 1

print("\nPress 'SPACE' to capture a photo. Press 'q' to quit.")

while True:
    ret, frame = cap.read() # 'frame' holo amader original, clean image
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # <<< CHANGE: Ekti copy toiri korchi shudhu dekhanor jonno
    display_frame = frame.copy() 

    # <<< CHANGE: Lekhagulo original 'frame'-e na, 'display_frame'-e add korchi
    cv2.putText(display_frame, "Press 'SPACE' to capture photo", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Count: {count-1}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # <<< CHANGE: User-ke 'display_frame' (lekha-shombolito) dekhacchi
    cv2.imshow("Static Image Capture", display_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == 32: 
        img_name = os.path.join(sign_path, f"{sign_name}_{count}.jpg")
        
        # <<< NO CHANGE: Ekhane amra original 'frame' (lekha chhara) save korchi
        cv2.imwrite(img_name, frame) 
        
        print(f"Captured: {img_name}")
        count += 1
        
cap.release()
cv2.destroyAllWindows()