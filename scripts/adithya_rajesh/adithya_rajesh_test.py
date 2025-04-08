import os

import cv2
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from bdgs.models.image_payload import ImagePayload
from bdgs.data.gesture import GESTURE
from scripts.common.get_learning_files import get_learning_files
from scripts.common.crop_image import crop_image
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH


def get_training_data():
    image_files = get_learning_files()
    processed_images = []
    labels = []

    alg = AdithyaRajesh()

    print("Processing images...")
    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        image = cv2.imread(image_path)
        label = int(image_file[1].split(" ")[0])
        cropped_image = crop_image(image, image_file[1])
        payload = ImagePayload(cropped_image)
        image = alg.process_image(payload)

        # The training process does not expect mini-batch dimention, so it's removed.
        image = np.squeeze(image)


        labels.append(label-1)
        processed_images.append(image)


    processed_images = np.array(processed_images)
    labels = np.array(labels)
    print("Processing images finished.")


    return processed_images, labels

def k_fold_train():
    images, labels = get_training_data()
    num_classes = len(GESTURE)

    acc_per_fold = []
    loss_per_fold = []
    fold_no = 0

    kfold = KFold(n_splits=5, shuffle=True)


    for train, test in kfold.split(images, labels):
        model = Sequential()
        model.add(Rescaling(1.0 / 255))


        #1st layer
        model.add(Conv2D(filters=8, kernel_size=(19, 19), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
        #2nd layer
        model.add(Conv2D(filters=16, kernel_size=(17, 17), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
        #3rd layer
        model.add(Conv2D(filters=32, kernel_size=(15, 15), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))


        model.add(Flatten())
        model.add(Dense(num_classes, activation="softmax"))


        model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
        )

        print(f'Training for fold {fold_no} ...')

        history = model.fit(images[train], labels[train],
                    batch_size=32,
                    epochs=2,
                    verbose="auto")

        scores = model.evaluate(images[test], labels[test], verbose=0)
        save_model(
            model=model,
            filepath=os.path.join(TRAINED_MODELS_PATH, "adithya_rajesh.keras")
        )

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1


def train():
    """Uses all available data to train."""
    images, labels = get_training_data()
    num_classes = len(GESTURE)

    model = Sequential()
    model.add(Rescaling(1.0 / 255))

    #1st layer
    model.add(Conv2D(filters=8, kernel_size=(19, 19), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
    #2nd layer
    model.add(Conv2D(filters=16, kernel_size=(17, 17), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
    #3rd layer
    model.add(Conv2D(filters=32, kernel_size=(15, 15), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))


    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))


    model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss=SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
    )


    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2,
                                                      random_state=42)

    # reduced the epochs from 20 to 3 to reduce overfitting for now.
    history = model.fit(x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=32,
                epochs=3,
                verbose="auto")

    save_model(
        model=model,
        filepath=os.path.join(TRAINED_MODELS_PATH, "adithya_rajesh.keras")
    )

def test_process_image():
    image_files = get_learning_files()

    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        image = cv2.imread(image_path)

        if image is not None:
            image_label = int(image_file[1].split(" ")[0])
            image = crop_image(image, image_file[1])
            alg = AdithyaRajesh()
            payload = ImagePayload(image=image)
            processed_image = alg.process_image(payload)
            # remove batch dimension (1, 100, 100, 3) -> (100, 100, 3)
            image_without_batch_dim = np.squeeze(processed_image)
            # set array datatype back to uint8
            image_without_batch_dim = np.astype(image_without_batch_dim, np.uint8)


            print(f"Image label: {image_label}")
            cv2.imshow("image", image_without_batch_dim)
            cv2.waitKey(2000)
        else:
            print(f"Failed to load image: {image_file}")


def classify_test():
    image_files = get_learning_files()

    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        image = cv2.imread(image_path)

        if image is not None:
            image_label = int(image_file[1].split(" ")[0])
            image = crop_image(image, image_file[1])

            alg = AdithyaRajesh()
            payload = ImagePayload(image)
            predicted_class, certainty = alg.classify(payload=payload)
            print(f"Correct class: {GESTURE(image_label).name}")
            print(f"Predicted class: {predicted_class}, certainty: {certainty}%")
            cv2.imshow("image", image)
            cv2.waitKey(2000)

        else:
            print(f"Failed to load image: {image_file}")

def evaluate():
    images, labels = get_training_data()
    model = load_model(os.path.join(TRAINED_MODELS_PATH, "adithya_rajesh.keras"))
    model.evaluate(images, labels, batch_size=32)


#k_fold_train()
#train()
#classify_test()
#test_process_image()
evaluate()
