import os
import cv2
import pandas
from sklearn.linear_model import LogisticRegression
from facial_features import detect_facial_features, distances_between_facial_features

if __name__ == '__main__':
    labels_file = '../../cleaned dataset/labels.csv'
    images_dir = '../../aligned dataset'
    print(os.path.exists(images_dir))
    labels_data = pandas.read_csv(labels_file, header=None)
    labels = {}
    label_idx = 0
    labeled_images = {}
    print("processing labels...")
    for i, row in labels_data.iterrows():
        image_name = row[0]
        label = row[1]
        if label not in labels.keys():
            labels[label] = label_idx
            label_idx += 1
        labeled_images[image_name] = labels[label]

    print(labels)

    # features
    print('extracting features...')
    train_data = []
    train_labels = []
    for image_name in os.listdir(images_dir):
        full_image_name = images_dir + '/' + image_name
        image = cv2.imread(full_image_name, cv2.IMREAD_GRAYSCALE)
        try:
            facial_features = detect_facial_features(image, True)
            distances = distances_between_facial_features(facial_features)
            train_data.append(distances)
            train_labels.append(labeled_images[image_name])
        except Exception:
            print("Error finding features in %s" % full_image_name)

    print("Total data: %d" % len(train_data))
    classifier = LogisticRegression()
    classifier.fit(train_data, train_labels)
