import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection()


pTime = 0
while True:
    success, img = capture.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceDetection.process(imgRGB)

    if results.detections:

        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) # this is just using the drawing_utilss

            #going for some manual approach
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
            cv2.rectangle(img, bbox, (255, 255, 0), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(1)