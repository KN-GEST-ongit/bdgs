import cv2
import numpy as np
import keras
import os

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from scripts.common.vars import TRAINED_MODELS_PATH


class Maung(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        # Experimental
        if processing_method != PROCESSING_METHOD.DEFAULT:
            from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
            return ALGORITHM_FUNCTIONS[processing_method].process_image(payload)

        gray = cv2.cvtColor(payload.image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.medianBlur(gray, 15)

        resized = cv2.resize(blurred, (140, 150))

        roberts_x = np.array([[1, 0], [0, -1]])

        roberts_y = np.array([[0, 1], [-1, 0]])

        dx = cv2.filter2D(resized, cv2.CV_64F, roberts_x)
        dy = cv2.filter2D(resized, cv2.CV_64F, roberts_y)

        gradient_orientation = np.arctan2(dy, dx)

        gradient_orientation_degrees = np.degrees(gradient_orientation) % 90

        hist, _ = np.histogram(gradient_orientation_degrees, bins=3, range=(0, 90))

        # return gradient_orientation_degrees
        return hist.astype(np.float32)

    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        predicted_class = 1
        certainty = 0
        model = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'maung.keras'))
        processed_image = self.process_image(payload=payload, processing_method=processing_method)
        processed_image = np.expand_dims(processed_image, axis=0)  #

        predictions = model.predict(processed_image)

        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction) + 1
            certainty = int(np.max(prediction) * 100)

        return GESTURE(predicted_class), certainty
