import cv2
import os
import mediapipe as mp
import numpy as np

DATASET_PATH = "dataset"
OUTPUT_PATH = "landmarks"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

labels = os.listdir(DATASET_PATH)

for label in labels:
    label_path = os.path.join(DATASET_PATH, label)
    output_file = os.path.join(OUTPUT_PATH, f"{label}.npy")

    data = []

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                data.append(landmarks)

    np.save(output_file, data)
    print(f"{label} done!")

print(" Landmark dataset created!")