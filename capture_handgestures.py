import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# Gesture Name
gesture_name = input("Enter Gesture Name: ")  # Type the name of the gesture
data = []

print("Collecting data for gesture:", gesture_name)
print("Press 's' to save frame, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            h, w, _ = frame.shape
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save data
                data.append(landmarks)
                print(f"Saved {len(data)} samples")

    cv2.imshow("Collecting Hand Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
        break

cap.release()
cv2.destroyAllWindows()

# Save data to CSV
df = pd.DataFrame(data)
df['label'] = gesture_name  # Add gesture label
df.to_csv("gesture_data.csv", mode='a', index=False, header=False)  # Append data
print("Data saved to 'gesture_data.csv'")
