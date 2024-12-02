import cv2
import os
from temp import detect_hand_box  # Import the function from temp.py
import mediapipe as mp

# Initialize MediaPipe Hands outside the function to reuse the graph
mp_hands = mp.solutions.hands


# Capture video from webcam
cap = cv2.VideoCapture(0)
image_count = 0

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)

data_type = int(input("Enter 1 for enlarging write data, 2 for move and 3 for clear data : "))

# Create a folder for saving hand images
if data_type == 1:
    output_folder = "write_images"
elif data_type == 2:
    output_folder = "move_images"
elif data_type == 3:
    output_folder = "clear_images"
else:
    print("Enter a number between 1 and 3")

os.makedirs(output_folder, exist_ok=True)

while cap.isOpened() and image_count<800:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Detect hand and get bounding box
    box = detect_hand_box(frame)
    if box:
        xmin, ymin, xmax, ymax = box

        # Ensure coordinates are within frame bounds
        height, width, _ = frame.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        # Crop the region containing the hand
        hand_image = frame[ymin:ymax, xmin:xmax]

        # Check if hand_image is valid (not empty)
        if hand_image.size > 0:
            # Save the cropped image
            image_path = os.path.join(output_folder, f"hand_{image_count}.jpg")
            cv2.imwrite(image_path, hand_image)
            print(f"Saved hand image at {image_path}")
            image_count += 1
        else:
            print("Cropped hand image is empty, skipping save.")

    # Display the output frame with the detected bounding box (optional)
    if box:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow("Hand Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()
