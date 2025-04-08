from numpy import ndarray

from bdgs.data.algorithm import ALGORITHM
from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload


def process_image(algorithm: ALGORITHM, payload: ImagePayload,
                  processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    processed = classifier.process_image(payload, processing_method)

    return processed


def classify(algorithm: ALGORITHM, payload: ImagePayload,
             processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction, certainty = classifier.classify(payload, processing_method)

    return prediction, certainty
