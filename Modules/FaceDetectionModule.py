import cv2
import mediapipe as mp
import copy



class FaceDetector():
    def __init__(self,
                 mode=False,
                 max_num_faces=2,
                 detectionCon=0.5,
                 trackingCon=0.5):
        self.mode = mode
        self.max_num_faces = max_num_faces
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detect = mp.solutions.face_detection

        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=3,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

        self.face_detection = self.mp_face_detect.FaceDetection()

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_detections(self, img):
        new_img = copy.copy(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)

        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(new_img, detection, self.drawing_spec)
        return new_img


    def find_face_mesh(self, img):
        new_img = copy.copy(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(new_img, face_landmarks,
                                        self.mp_face_mesh.FACE_CONNECTIONS,
                                        self.drawing_spec, self.drawing_spec)
        return new_img


def main():
    cap = cv2.VideoCapture(1)
    face_detector = FaceDetector()
    while True:
        _, img = cap.read()
        img_rect = face_detector.find_face_detections(img)
        img_mesh = face_detector.find_face_mesh(img)

        cv2.imshow('rect', img_rect)
        cv2.imshow('mesh', img_mesh)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()