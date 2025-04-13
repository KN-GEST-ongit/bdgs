import os
import cv2
import numpy as np
import tensorflow as tf
import keras

from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH
from bdgs.classifier import process_image
from scripts.common.crop_image import crop_image
from bdgs.models.image_payload import ImagePayload
from bdgs.data.algorithm import ALGORITHM
from sklearn.model_selection import train_test_split


def learn():
    processed_images = []
    etiquettes = []
    images = get_learning_files(limit=1000, shuffle=True)
    for image, hand_recognition_data, _ in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image))
        hand_image = crop_image(cv2.imread(image_path), hand_recognition_data)
        processed_image = process_image(
            algorithm=ALGORITHM.MAUNG,
            payload=ImagePayload(image=hand_image)
        )
        processed_images.append(processed_image)
        etiquettes.append(int(hand_recognition_data.split(" ")[0]) - 1)

    X = np.array(processed_images)
    y = np.array(etiquettes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    num_classes = len(np.unique(y))
    input_length = X.shape[1]
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_length,)),
        tf.keras.layers.Dense(num_classes, activation='hard_sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    keras.models.save_model(model, os.path.join(TRAINED_MODELS_PATH, 'maung.keras'))


if __name__ == "__main__":
    learn()
