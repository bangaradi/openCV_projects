import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, staticImg=False, maxFaces=2, refineLm=False, detectionCon=0.5, trackCon=0.5):
        self.staticImg = staticImg
        self.maxFaces = maxFaces
        self.refineLm = refineLm
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def FindFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:

            for id, landmarks in enumerate(self.results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(landmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])

                faces.append([face])

        return img, faces






def main():
    capture = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    pTime = 0
    while True:
        success, img = capture.read()

        img, faces = detector.FindFaceMesh(img)

        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'fps: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()