import cv2
import mediapipe as mp
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from temp import detect_hand_box
from PIL import Image
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8')

# Load the trained model
gesture_classifier = load_model('gesture_classification_model.h5')

# Function to preprocess a new image for prediction
def preprocess_image(img):
    img = Image.fromarray(img)  # Convert the NumPy array to a PIL image
    img = img.resize((128, 128))  # Resize to match the training input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to match the training data
    return img_array


# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

middle_finger_positions = [[], []]
prev_mode = 0
# Variable to store the mode when we need to clear the path
clear_lines = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert the frame color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw hand landmarks and display positions
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            
            xmin, ymin, xmax, ymax = detect_hand_box(frame)
            height, width, _ = frame.shape
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)
            mini_frame = frame[ymin:ymax, xmin:xmax]

            # Preprocess the mini frame for prediction
            procecced_mini_frame = preprocess_image(mini_frame)
            curr_mode_matrix = gesture_classifier.predict(procecced_mini_frame)
            
            # Debug: Print current mode matrix and detected mode
            print("Current Mode Matrix:", curr_mode_matrix)
            curr_mode = np.argmax(curr_mode_matrix)
            print("Predicted Mode:", curr_mode)
        
            
            if curr_mode == 2:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the position of the middle finger (landmark 12)
                height, width, _ = frame.shape
                middle_x, middle_y = int(hand_landmarks.landmark[12].x * width), int(hand_landmarks.landmark[12].y * height)
                
                if prev_mode == 1:
                    middle_finger_positions.append([])

                middle_finger_positions[-1].append((middle_x, middle_y))
                
                for i in middle_finger_positions:
                    for j in range(1, len(i)):
                        cv2.line(frame, i[j-1], i[j], (0, 0, 255), 2)
                 
            elif curr_mode == 1:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            elif curr_mode == 0:
                # Clear the lines (reset the finger positions)
                middle_finger_positions = [[], []]  # This clears the path of the middle finger
                clear_lines = True
                
            prev_mode = curr_mode
            if prev_mode == 1:
                middle_finger_positions.append([])

    # If clear_lines is True, reset the state of the lines on the frame
    if clear_lines:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR to avoid any color space issues
        clear_lines = False  # Reset the flag after clearing lines

    # Save the current frame periodically (every iteration)
    cv2.imwrite("current_frame.jpg", frame)  # Save the current frame as 'current_frame.jpg'

    # Display the output frame
    cv2.imshow('Finger Position Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()
