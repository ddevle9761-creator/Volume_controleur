import cv2
import numpy as np
from main import HandDetector
import subprocess

class VolumeControleur:

    @staticmethod
    def set_volume(volume):
        subprocess.run(["pactl", "set-sink-volume",
                        "@DEFAULT_SINK@", f"{volume}%"],
                       capture_output=True, text=True
                       )

    def draw(self):
        pass



cap = cv2.VideoCapture(0)

H, W = 60, 130

def lancer() :
    pTime = 0
    distance = 0
    volume = 0
    detector = HandDetector()
    game = VolumeControleur()

    while True :
        success, img = cap.read()

        cv2.flip(img, 1)
        img = detector.findHands(img, draw=False)
        position_main = detector.position_pouce_index(img=img, draw=True)


        if position_main :
            for i in position_main:
                distance = i[-1]

                volume = np.interp(distance, [0, 60], [0, 100])


        game.set_volume(int(volume))

        hateur = np.interp(distance, [0, 100], [0, 250])

        cv2.rectangle(img, (10, 300), (80, 300 - int(hateur)),(0, 255, 0), -1)

        cv2.line(img, (10, 302), (81, 302), (0, 0, 255), 1)
        cv2.line(img, (10, 51), (81, 51), (0, 0, 255), 1)
        cv2.line(img, (10, 51), (10, 302), (0, 0, 255), 1)
        cv2.line(img, (81, 51), (81, 302), (0, 0, 255), 1)



        cv2.putText(img, f"{int(volume)}%", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (122, 255, 0), 2)

        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    lancer()

    cap.release()
    cv2.destroyAllWindows()



