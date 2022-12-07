import cv2
import time
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 10
eraserThickness = 30

folderPth = "Images"
myList = os.listdir(folderPth)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPth}/{imPath}')
    image = cv2.flip(image, 1)
    overlayList.append(image)
# print(overlayList)
paintBar = overlayList[0]
drawColor = (255, 0, 255)

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detector = htm.handDetector(detectionCon=0.8)

xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), dtype=np.uint8)
imgCanvas = cv2.flip(imgCanvas, 1)

while True:
    # taking the image
    success, img = capture.read()

    #finding the hand landmarks
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        #landmarks of the fingertips
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #checking which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        #selection mode setting, 2 fingers are up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1-15), (x2, y2+15), drawColor, cv2.FILLED)
            xp, yp = 0, 0
            if (y1 < 125):
                if (100 < x1 < 340):
                    paintBar = overlayList[1]
                    drawColor = (0, 0, 0)
                elif (340 < x1 < 500):
                    paintBar = overlayList[0]
                    drawColor = (53, 245, 47)
                elif (500 < x1 < 640):
                    paintBar = overlayList[3]
                    drawColor = (197, 47, 245)
                elif (640 < x1 < 780):
                    paintBar = overlayList[2]
                    drawColor = (47, 225, 245)
                elif (800 < x1 < 950):
                    paintBar = overlayList[4]
                    drawColor = (255, 0, 0)

        #drawing mode, index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print('draw mode')
            if xp==0 and yp==0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (x1, y1), (xp, yp), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (x1, y1), (xp, yp), drawColor, brushThickness)

            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, ImgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    ImgInv = cv2.cvtColor(ImgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, ImgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = paintBar
    img = cv2.flip(img, 1)

    cv2.imshow("img", img)
    # imgshowing = cv2.flip(imgCanvas, 1)
    # cv2.imshow('canvas', imgshowing)
    cv2.waitKey(1)
