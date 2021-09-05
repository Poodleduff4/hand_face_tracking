import cv2
import mediapipe as mp
from mediapipe.python import solutions

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detect = mp.solutions.face_detection

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(1)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

face_detection = mp_face_detect.FaceDetection()

while True:
    _, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_RGB)
    # results = face_detection.process(img_RGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(img, face_landmarks,
                                      mp_face_mesh.FACE_CONNECTIONS,
                                      drawing_spec, drawing_spec)

    cv2.imshow('image', img)
    cv2.waitkey(5)