import numpy as np
import cv2

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE


class AdithyaRajesh(BaseAlgorithm):
    def process_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        image = image.astype(np.float32)
        #expand to get shape (1, 100, 100, 3)
        image = np.expand_dims(image, axis=0)

        return image

    def classify(self, image) -> GESTURE:
        return GESTURE.GOODBYE
