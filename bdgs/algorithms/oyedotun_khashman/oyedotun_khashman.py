import cv2
import numpy as np
from numpy import ndarray
from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData

def extract_hand(image: ndarray):
    binary_image = image.astype(np.uint8)

    ys, xs = np.where(binary_image > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    return image[y_min:y_max + 1, x_min:x_max + 1]


class OyedotunKhashman(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
        gray_image = cv2.cvtColor(payload.image, cv2.COLOR_BGRA2GRAY)
        _, binary_threshed = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        # authors specieid the size of median filter (15 x 10)
        median_filtered = cv2.medianBlur(binary_threshed, 13)
        extracted_hand = extract_hand(median_filtered)
        resized = cv2.resize(extracted_hand, (32, 32), interpolation=cv2.INTER_AREA)

        return resized
    


    def classify(self, payload: ImagePayload, custom_model_path = None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> GESTURE:
        raise NotImplementedError("Method classify not implemented")

    def learn(self, learning_data: list[LearningData], target_model_path: str) -> (float, float):
        raise NotImplementedError("Method learn not implemented")
        
