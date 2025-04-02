from enum import StrEnum
from typing import Any

from cv2 import Mat
from numpy import ndarray, dtype

from bdgs.algorithms.maung.maung import Maung
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.gesture import GESTURE


class ALGORITHM(StrEnum):
    MURTHY_JADON = "MURTHY_JADON"
    MAUNG = "MAUNG"


ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.MAUNG: Maung(),
}


def recognize(image: Mat | ndarray[Any, dtype], algorithm: ALGORITHM) -> GESTURE:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier.classify(image)

    return prediction
