import Modules.HandTrackingModule as htm
import Modules.FaceDetectionModule as fdm
import cv2



cap = cv2.VideoCapture(1)
hand_detector = htm.handDetector()
face_detector = fdm.FaceDetector()

while True:
    _, img = cap.read()
    img = hand_detector.find_hands(img)
    img = face_detector.find_face_mesh(img)

    cv2.imshow('face and hands', img)
    cv2.waitKey(5)