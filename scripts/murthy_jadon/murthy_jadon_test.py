import os

import cv2

from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from scripts.common.get_learning_files import get_learning_files

folder_path = os.path.abspath("../../../bdgs_photos")


def test_process_image():
    images = get_learning_files()

    for image_file in images:
        if image_file[1].split(" ")[0] == "0":
            continue

        image_path = os.path.join(folder_path, image_file[0])
        bg_image_path = os.path.join(folder_path, image_file[2])

        hand_image = cv2.imread(image_path)
        background_image = cv2.imread(bg_image_path)

        # cv2.imshow("bg_image", background_image)
        # cv2.imshow("hand_image", hand_image)

        algorithm = MurthyJadon()
        processed_image = algorithm.process_image(
            payload=MurthyJadonPayload(image=hand_image, bg_image=background_image))
        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_process_image()
