# dynamic_data_collector.py

# Import necessary libraries
import cv2
import os
import time

# --- Logic for DYNAMIC data collection (Fully Automated with Countdown) ---
DATA_PATH = "Dynamic_Data"
# --- The countdown duration in seconds ---
COUNTDOWN_SECONDS = 1

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

sign_name = input("Enter the dynamic sign name: ")
total_videos_to_collect = int(input(f"How many NEW videos to collect for '{sign_name}': "))
record_duration = float(input("Enter recording duration per video in seconds (e.g., 2.5, 3, 4): "))

sign_path = os.path.join(DATA_PATH, sign_name)
if not os.path.exists(sign_path):
    os.makedirs(sign_path)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nPosition yourself. Press 's' ONCE to start the entire batch recording.")

# Wait for the initial 's' key press
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "Press 's' to start the batch...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Fully Automated Capture", frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key & 0xFF == ord('s'):
        break
        
start_num = len(os.listdir(sign_path)) + 1
end_num = start_num + total_videos_to_collect - 1

for video_num in range(start_num, end_num + 1):
    # --- MODIFICATION: Added a visible numerical countdown ---
    print(f"\nGet ready for video #{video_num}.")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        # Read a fresh frame for a live background
        ret, frame = cap.read() 
        if not ret:
            break
        # Define font size and thickness for the countdown number
        font_scale = 4
        thickness = 5
        text = str(i)
        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = int((frame_width - text_width) / 2)
        text_y = int((frame_height + text_height) / 2)
        
        # Display the countdown number on the screen
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv2.imshow("Fully Automated Capture", frame)
        # Wait for 1 second
        cv2.waitKey(1000)

    # Start recording immediately after countdown
    print(f"Recording video #{video_num} for {record_duration} seconds...")
    
    start_time = time.time()
    video_filename = os.path.join(sign_path, f"{sign_name}_{video_num}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))

    while (time.time() - start_time) < record_duration:
        ret_rec, rec_frame = cap.read()
        if not ret_rec:
            break
        
        out.write(rec_frame)
        cv2.putText(rec_frame, "RECORDING...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Fully Automated Capture", rec_frame)
        cv2.waitKey(1)

    out.release()
    print(f"Saved: {video_filename}")

print("\nBatch recording complete!")
cap.release()
cv2.destroyAllWindows()