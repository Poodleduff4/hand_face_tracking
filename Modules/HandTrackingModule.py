import mediapipe as mp
import cv2
import numpy as np
import math
import copy


class handDetector():
    def __init__(self,
                 mode=False,
                 maxHands=7,
                 detectionCon=0.7,
                 trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands,
                                         self.detectionCon, self.trackingCon)

        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        new_img = copy.copy(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # blackImg = np.zeros(img.shape)

        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLmk in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        new_img, handLmk, self.mp_hands.HAND_CONNECTIONS)

        return new_img

    def find_position(self, img, handNum=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                lmList.append([id, cx, cy])
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        _, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img, draw=False)
        if len(lmList) != 0:
            # print(lmList)
            pass

        cv2.imshow('image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()