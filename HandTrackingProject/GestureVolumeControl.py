import cv2
import time
import numpy
import HandTrackingModule as htm
import math
import osascript



#setting camera height and width
W_CAM, H_CAM = 1280, 720

capture = cv2.VideoCapture(0)
capture.set(3, W_CAM)
capture.set(4, H_CAM)

detector = htm.handDetector(detectionCon = 0.7)


pTime = 0
while True:
    success, img = capture.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        set_vol = length/500 * 100
        # print(set_vol)
        vol = 'set volume output volume ' + str(set_vol)
        osascript.osascript(vol)
        # osascript.run(vol)
        # osascript.

        vol_result = osascript.osascript("get volume settings")
        # print(vol_result)
        volInfo = vol_result[1].split(',')
        outputVol = volInfo[0].replace('output volume:', '')
        outputVol = int(outputVol)
        # print(f'outputVol : {outputVol}')
        outputVol = 500 - 5*outputVol
        cv2.line(img, (50, 500), (50, outputVol), (255, 255, 255), 20)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'fps: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(1)