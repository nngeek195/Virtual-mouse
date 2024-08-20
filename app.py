import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Distance threshold for click detection
CLICK_THRESHOLD = 30  # Adjust this value based on your needs
# Debounce time in seconds
DEBOUNCE_TIME = 0.2

last_click_time = 0

# Variables to store the previous mouse position for smoothing
prev_mouse_x, prev_mouse_y = pyautogui.position()
SMOOTHING_FACTOR = 0.5  # Adjust this value for more or less smoothing

def draw_axes(frame, landmarks, w, h):
    index_finger_tip = landmarks[8]
    middle_finger_tip = landmarks[12]
    cx1, cy1 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
    cx2, cy2 = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

    # Draw axes
    cv2.line(frame, (0, cy1), (w, cy1), (0, 255, 0), 2)  # Horizontal line for index finger
    cv2.line(frame, (cx1, 0), (cx1, h), (0, 255, 0), 2)  # Vertical line for index finger
    cv2.line(frame, (0, cy2), (w, cy2), (255, 0, 0), 2)  # Horizontal line for middle finger
    cv2.line(frame, (cx2, 0), (cx2, h), (255, 0, 0), 2)  # Vertical line for middle finger

    # Draw virtual line between the tips of the index finger and middle finger
    cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)  # Yellow line for virtual connection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            draw_axes(frame, landmarks, w, h)

            # Use pyautogui to move mouse with smoothing
            index_finger_tip = landmarks[8]
            screen_width, screen_height = pyautogui.size()
            screen_x = screen_width / w * int(index_finger_tip.x * w)
            screen_y = screen_height / h * int(index_finger_tip.y * h)
            
            # Smooth the mouse movement
            smooth_x = prev_mouse_x * (1 - SMOOTHING_FACTOR) + screen_x * SMOOTHING_FACTOR
            smooth_y = prev_mouse_y * (1 - SMOOTHING_FACTOR) + screen_y * SMOOTHING_FACTOR
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

            # Check distance between thumb tip and index finger tip for single click
            thumb_tip = landmarks[4]
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            distance_thumb_index = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Check distance between index finger tip and middle finger tip for double click
            middle_finger_tip = landmarks[12]
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            distance_index_middle = math.hypot(index_x - middle_x, index_y - middle_y)

            # Perform single click if the distance is below the threshold and debounce time has passed
            current_time = time.time()
            if distance_thumb_index < CLICK_THRESHOLD and (current_time - last_click_time) > DEBOUNCE_TIME:
                pyautogui.click()
                last_click_time = current_time

            # Perform double click if the distance between index and middle finger tips is below the threshold
            if distance_index_middle < CLICK_THRESHOLD and (current_time - last_click_time) > DEBOUNCE_TIME:
                pyautogui.doubleClick()
                last_click_time = current_time

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
