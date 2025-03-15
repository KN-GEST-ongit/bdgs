from enum import StrEnum
from typing import Any

from cv2 import Mat
from numpy import ndarray, dtype

from bdgs.algorithms.alg_2.alg_2 import Alg2
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from bdgs.gesture import GESTURE


class ALGORITHM(StrEnum):
    MURTHY_JADON = "MURTHY_JADON"
    ADITHYA_RAJESH = "ADITHYA_RAJESH"


ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.ADITHYA_RAJESH: AdithyaRajesh(),
}


def recognize(image: Mat | ndarray[Any, dtype], algorithm: ALGORITHM) -> GESTURE:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier.classify(image)

    return prediction
