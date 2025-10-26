import os
import pickle

import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import ndarray

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData

def extract_features(image: ndarray) -> np.array:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, w, _ = hsv.shape
    # values of n, m were not specified in the paper. to be determined with experiments.
    n, m = 4, 4
    block_h = h // n
    block_w = w // m

    features = []
    for i in range(n):
        for j in range(m):
            block = hsv[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]            
            hist = cv2.calcHist([block], [0, 1, 2], None, [8, 8, 8],
                                 [0, 180, 0, 256, 0, 256])

            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

    return np.array(features)
    

class NaidooOmlin(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
        raise NotImplementedError("Naidoo_Omlin method does not process image.")

    def classify(self, payload: ImagePayload, custom_model_path=None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> GESTURE:
        return GESTURE.LIKE, 100

    def learn(self, learning_data: list[LearningData], target_model_path: str) -> (float, float):
        labels = []
        features = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)
            labels.append(data.label.value - 1)
            features.append(extract_features(hand_image))

        X = np.array(features)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        model_path = os.path.join(target_model_path, 'naidoo_omlin.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        return acc, 1 - acc
