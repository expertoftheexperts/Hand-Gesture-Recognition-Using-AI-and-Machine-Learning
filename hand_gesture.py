import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("gesture_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

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
            landmarks = {id: (int(lm.x * w), int(lm.y * h)) for id, lm in enumerate(hand_landmarks.landmark)}

            # Check finger positions for gesture recognition
            fingers = []
            for tip in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky finger tips
                if landmarks[tip][1] < landmarks[tip - 2][1]:  # If tip is above knuckle
                    fingers.append(1)  # Finger is open
                else:
                    fingers.append(0)  # Finger is closed

            # Check thumb separately
            thumb = 1 if landmarks[4][0] > landmarks[2][0] else 0  # Thumb direction check

            # Gesture Recognition Logic
            gesture = "Unknown"
            
            if fingers == [1, 1, 1, 1] and thumb == 1:
                gesture = "Open Palm ✋"
            elif fingers == [0, 0, 0, 0] and thumb == 0:
                gesture = "Fist ✊"
            elif fingers == [0, 0, 0, 0] and thumb == 1:
                gesture = "Thumbs Up 👍"
            elif fingers == [1, 1, 0, 0] and thumb == 0:
                gesture = "Victory ✌️"
            elif fingers == [1, 0, 0, 0] and thumb == 1:
                gesture = "OK 👌"
            elif fingers == [1, 0, 0, 1] and thumb == 0:
                gesture = "Rock 🤘"
            elif fingers == [1, 1, 1, 1] and thumb == 0:
                gesture = "Stop ✋"
            elif fingers == [1, 0, 0, 0] and thumb == 0:
                gesture = "Pointing Up ☝️"
            elif fingers == [1, 1, 1, 1] and thumb == 1 and abs(landmarks[4][0] - landmarks[8][0]) < 30:
                gesture = "Namaste 🙏"
            elif fingers == [1, 1, 1, 1] and thumb == 1 and landmarks[4][1] > landmarks[8][1]:
                gesture = "Salaam 🤲"
            elif fingers == [1, 1, 1, 1] and thumb == 1 and abs(landmarks[4][0] - landmarks[20][0]) > 80:
                gesture = "Waving 👋"
            elif fingers == [1, 0, 0, 0] and thumb == 1 and fingers[3] == 1:
                gesture = "Call Me 🤙"

            # Display gesture on screen
            cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
