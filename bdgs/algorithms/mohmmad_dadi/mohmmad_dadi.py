import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload


class MohmmadDadi(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        pass

    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        pass
