import cv2
import time
import numpy as np
import Modules.HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.7, trackingCon=0.8)

while True:
    _, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)


    if len(lmList) != 0:
        pt1 = (lmList[4][1], lmList[4][2])
        pt2 = (lmList[8][1], lmList[8][2])

        cv2.circle(img, pt1, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pt2, 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, pt1, pt2, (255, 0, 255), 2)

        x_m_point = int((pt1[0] + pt2[0]) / 2)
        y_m_point = int((pt1[1] + pt2[1]) / 2)
        cv2.circle(img, (x_m_point, y_m_point), 10, (255, 0, 0), cv2.FILLED)

        distance = int(math.sqrt(math.pow(lmList[4][1] - lmList[8][1], 2) + math.pow(lmList[4][2] - lmList[8][2], 2)))

        # hand range = ~0 - ~100
        

        # print(distance)
        if distance < 10:
            cv2.circle(img, (x_m_point, y_m_point), 10, (0, 255, 0), cv2.FILLED)

        volRange = volume.GetVolumeRange()
        minVol = volRange[0]
        maxVol = volRange[1]

        vol = np.interp(distance, [0, 80], [minVol, maxVol])
        print('vol: ' + str(vol))


        volume.SetMasterVolumeLevel(vol, None)


    

    cv2.imshow('image', img)
    cv2.waitKey(1)


    