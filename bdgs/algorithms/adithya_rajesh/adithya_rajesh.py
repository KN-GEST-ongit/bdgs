import cv2
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload


class AdithyaRajesh(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        image = payload.image
        image = cv2.resize(image, (100, 100))
        image = image.astype(np.float32)
        # expand to get shape (1, 100, 100, 3)
        image = np.expand_dims(image, axis=0)

        return image

    def classify(self, image, processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        return GESTURE.TEN, 100
