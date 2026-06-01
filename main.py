import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleSource]
import sys
import os


# FIX FOR EXE PATHS

base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))


# LOAD MODEL

model_path = os.path.join(base_path, "model/sign_model.h5")
labels_path = os.path.join(base_path, "model/labels.npy")

if not os.path.exists(model_path):
    print(" Model file not found:", model_path)
    exit()

if not os.path.exists(labels_path):
    print(" Labels file not found:", labels_path)
    exit()

model = load_model(model_path)
labels = np.load(labels_path)

print(" Model loaded successfully")


# MEDIAPIPE

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


# CAMERA 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(" Camera not detected")
    exit()
else:
    print(" Camera started successfully")


# VARIABLES

current_text = ""
stable_char = ""
count = 0

CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 15

print(" Starting main loop...")


# TEXT WRAP FUNCTION

def draw_multiline_text(img, text, x, y, max_width, line_height):
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        (w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        if w < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "

    lines.append(current_line)

    for i, line in enumerate(lines[-3:]):
        cv2.putText(img, line.strip(), (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 2)


# MAIN LOOP

while True:
    ret, frame = cap.read()

    if not ret:
        print(" Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    display_text = "No Hand"
    confidence_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:
                input_data = np.array(landmarks).reshape(1, -1)

                prediction = model.predict(input_data, verbose=0)
                confidence = np.max(prediction)
                predicted_class = labels[np.argmax(prediction)]

                if confidence > CONFIDENCE_THRESHOLD:
                    display_text = predicted_class
                    confidence_text = f"{confidence:.2f}"

                    if predicted_class == stable_char:
                        count += 1
                    else:
                        stable_char = predicted_class
                        count = 0

                    if count > STABILITY_FRAMES:
                        current_text += predicted_class
                        count = 0
                else:
                    display_text = "Low Confidence"

    
    # UI CANVAS

    canvas = np.zeros((650, 1100, 3), dtype=np.uint8)

    for i in range(650):
        shade = int(15 + (i / 650) * 40)
        canvas[i, :] = (shade, shade + 5, shade + 10)

    cam = cv2.resize(frame, (650, 480))
    canvas[80:560, 50:700] = cam

    cv2.putText(canvas, "SIGN LANGUAGE TRANSLATOR", (200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.rectangle(canvas, (750, 120), (1050, 350), (200, 200, 200), 2)
    cv2.putText(canvas, "OUTPUT", (860, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    draw_multiline_text(canvas, current_text, 770, 180, 260, 40)

    cv2.putText(canvas, f"Prediction: {display_text}", (50, 610),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)

    cv2.putText(canvas, f"Confidence: {confidence_text}", (350, 610),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.putText(canvas, "C: Clear | SPACE: Add Space | Q: Quit", (650, 610),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Sign Language Translator", canvas)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        current_text = ""
    elif key == 32:
        if len(current_text) == 0 or current_text[-1] != " ":
            current_text += " "

# CLEANUP
cap.release()
cv2.destroyAllWindows()