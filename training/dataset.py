import os
import cv2
import csv
import numpy as np
from keras.utils import np_utils

LABELS_MAPPING = {
    'Fear': 0,
    'Disgust': 1,
    'Surprise': 2,
    'Contempt': 3,
    'Anger': 4,
    'Neutral': 5,
    'Happiness': 6,
    'Sadness': 7
}


def read_data(labels_file_path, images_dir, num_classes, num_images=1000000):
    images = list()
    labels = list()

    image_labels = {}

    with open(labels_file_path, 'r') as f:

        reader = csv.DictReader(f, fieldnames=['image', 'label'])

        for row in reader:
            image_labels[row['image']] = row['label']

    ctr = 0
    for image_name in os.listdir(images_dir):
        if image_name in image_labels:
            image_path = os.path.join(images_dir, image_name)

            img = cv2.imread(image_path, 0)
            # img = cv2.imread(image_path, 1)

            # opencv returns greyscale images in 2d arrays and I need them in 3d, so I convert it to a 3d array
            img = np.atleast_3d(img)

            images.append(img)
            labels.append(LABELS_MAPPING[image_labels[image_name]])
            ctr += 1
            if ctr >= num_images:
                break

    images = np.array(images, dtype=object)
    labels = np.array(labels, dtype=object)
    labels = np_utils.to_categorical(labels, num_classes)

    return images, labels