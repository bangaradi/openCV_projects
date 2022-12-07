import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius=1)
pTime = 0
while True:
    success, img = capture.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for id, landmarks in enumerate(results.multi_face_landmarks):
            mpDraw.draw_landmarks(img, landmarks, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'fps: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)