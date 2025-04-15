import os

import cv2
import numpy as np

from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson import IslamHossainAndersson
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.gesture import GESTURE
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from scripts.common.camera_test import camera_test
from scripts.common.crop_image import crop_image
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def test_process_image():
    image_files = get_learning_files()

    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        bg_image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))
        background = cv2.imread(str(bg_image_path))
        image = cv2.imread(str(image_path))

        if image is not None:
            image_label = int(image_file[1].split(" ")[0])
            image = crop_image(image, image_file[1])
            background = crop_image(background, image_file[1])
            alg = IslamHossainAndersson()
            payload = IslamHossainAnderssonPayload(image=image, bg_image=background)
            processed_image = alg.process_image(payload)

            print(f"Image label: {image_label}")
            cv2.imshow("image", processed_image)
            cv2.waitKey(2000)
        else:
            print(f"Failed to load image: {image_file}")


if __name__ == "__main__":
    test_process_image()
    # classify_test()
    #cam_test()
