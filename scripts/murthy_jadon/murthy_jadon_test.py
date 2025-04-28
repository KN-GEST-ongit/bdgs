from bdgs.data.algorithm import ALGORITHM
from scripts.common.classification_test import classification_test

if __name__ == "__main__":
    # image_processing_test(ALGORITHM.MURTHY_JADON)
    classification_test(ALGORITHM.MURTHY_JADON)
    # camera_test(algorithm=ALGORITHM.MURTHY_JADON, show_prediction_tresh=60)
