from bdgs.data.algorithm import ALGORITHM
from scripts.common.classification_test import classification_test

if __name__ == "__main__":
    # image_processing_test(ALGORITHM.EID_SCHWENKER)
    classification_test(ALGORITHM.EID_SCHWENKER)
    # camera_test(algorithm=ALGORITHM.EID_SCHWENKER, show_prediction_tresh=60)
