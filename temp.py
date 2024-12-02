# temp.py
import cv2
import mediapipe as mp

def detect_hand_box(frame):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                height, width, _ = frame.shape
                xmin, xmax = int(min(x_coords) * width), int(max(x_coords) * width)
                ymin, ymax = int(min(y_coords) * height), int(max(y_coords) * height)
                return xmin, ymin, xmax, ymax
        return [0,0,640,480]
