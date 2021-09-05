import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blackImg = np.zeros(img.shape)

    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLmk in results.multi_hand_landmarks:
            for id, lm in enumerate(handLmk.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
            mp_drawing.draw_landmarks(img, handLmk, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('image', img)
    cv2.waitKey(1)