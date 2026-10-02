import time
import cv2
import mediapipe
import mediapipe as mp
import math

import numpy as np

print(mediapipe.__version__)


class HandDetector:
    def __init__(self, mode=False, maxHands= 2, detectionCon=0.5, trackCon=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHAnds = mp.solutions.hands  # initialiser les mains
        self.hands = self.mpHAnds.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon,

                                        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB, )

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHAnds.HAND_CONNECTIONS)

        return img

    def trouver_Position(self, img, handNo=0):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhands.landmark):
                 h, w, c = img.shape
                 cx, cy = int(lm.x * w), int(lm.y * h)
                 landmarkList.append([id, cx, cy])

        return landmarkList


    def position_pouce_index(self, img, draw=False, fing1=4, fing2=8):
        landmarkList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for main in self.results.multi_hand_landmarks:
                pouce = main.landmark[fing1] # le majeur
                pouce_x, pouce_y = int(pouce.x * w), int(pouce.y * h)

                index = main.landmark[fing2] # l'index
                index_x, index_y = int(index.x * w), int(index.y * h)


                distance = np.linalg.norm(np.array(pouce_x) - np.array(index_x))
                landmarkList.append([index_x, index_y,pouce_x, pouce_y, distance])


                if draw:
                    cv2.circle(img, (pouce_x, pouce_y), 8, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (index_x, index_y), 8, (255, 0, 0), cv2.FILLED)
                    #cv2.line(img, (pouce_x, pouce_y), (index_x, index_y), (255, 121, 143), 2)


        return landmarkList


    def mesure_la_main(self, img, draw=True):
        landmarkList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for main in self.results.multi_hand_landmarks:
                majeur = main.landmark[8]
                majeur_x, majeur_y = int(majeur.x * w), int(majeur.y * h)

                depart = main.landmark[12]
                depart_x, depart_y = int(depart.x * w), int(depart.y * h)

                print(depart_x, depart_y, majeur_x, majeur_y)

                distance = abs(majeur_x - depart_x)
                print(distance)

                cv2.line(img, (depart_x, depart_y),(majeur_x, majeur_y), (255, 121, 143), 2)

                landmarkList.append([depart_x, depart_y, majeur_x, majeur_y])
                if draw:
                    pass

        return landmarkList


    def index_doits(self):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            for main in self.results.multi_hand_landmarks:
                majeur = main.landmark
                for id, pos in enumerate(majeur):
                    landmarkList.append([id, pos.x, pos.y])


        return landmarkList

    def flux_ameliorer(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)

        adapt_thre = cv2.adaptiveThreshold(thresh, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY,
                                           9, 2)


        contour, _ = cv2.findContours(adapt_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            if 1000 < cv2.contourArea(cnt) < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img


H, W = 60, 130

def lancer() :
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True :
        success, img = cap.read()
        cv2.flip(img, 1)
        img = detector.findHands(img, draw=True)
        # landmarkList = detector.trouver_Position(img)
        # index_positon = detector.position_pouce_index(img, draw=True)
        #
        #
        # if len(landmarkList) != 0 :
        #     for m in landmarkList:
        #        pass
        # if len(index_positon) != 0 :
        #     for _ in index_positon:
        #         print(i)

        img = detector.flux_ameliorer(img)



        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fsp_text = "FPS: " + str(int(fps))


        cv2.putText(img, fsp_text, (10, 70),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,255),3)

        if not success :
            break




        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
if __name__ == '__main__':
    lancer()





