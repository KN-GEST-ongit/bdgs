import cv2
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE
from bdgs.models.image_payload import ImagePayload


class Maung(BaseAlgorithm):
    def process_image(self, payload: ImagePayload) -> np.ndarray:
        gray = cv2.cvtColor(payload.image, cv2.COLOR_BGR2GRAY)

        # ujednolicenie tła - rozmycie medianowe
        blurred = cv2.medianBlur(gray, 5)

        resized = cv2.resize(blurred, (140, 150))

        # detekcja krawędzi - operatora różnicowego
        dx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)

        # obliczanie orientacji gradientu
        gradient_orientation = np.arctan2(dy, dx)

        return gradient_orientation

    def classify(self, image) -> GESTURE:
        return GESTURE.OK
