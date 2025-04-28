from bdgs.data.algorithm import ALGORITHM
from scripts.common.classification_test import classification_test

if __name__ == "__main__":
    # image_processing_test(ALGORITHM.PINTO_BORGES)
    classification_test(ALGORITHM.PINTO_BORGES)
    # camera_test(algorithm=ALGORITHM.PINTO_BORGES, show_prediction_tresh=60)
