from numpy import ndarray
from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.algorithms.nguyen_huynh.nguyen_huynh_learning_data import NguyenHuynhLearningData
from bdgs.algorithms.nguyen_huynh.nguyen_huynh_payload import NguyenHuynhPayload
from bdgs.common.crop_image import crop_image
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData


class NguyenHuynh(BaseAlgorithm):
    def process_image(self, payload: NguyenHuynhPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
        image = payload.image
        coords = payload.coords
        if coords is not None:
            image = crop_image(image=image, coords=coords)

        return image

    def classify(self, payload: NguyenHuynhLearningData, custom_model_path = None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> GESTURE:
        raise NotImplementedError("Method classify not implemented")

    def learn(self, learning_data: list[LearningData], target_model_path: str) -> (float, float):
        raise NotImplementedError("Method learn not implemented")
        
