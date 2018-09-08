import os
import random

import pandas

# Full data-set
INPUT_DIRECTORY = '../../../aligned dataset'
LABELS_FILE = '../../../aligned dataset/updated_labels.csv'
# Data-set locations
TRAIN_DIRECTORY = '../../../aligned dataset/train'
VALIDATION_DIRECTORY = '../../../aligned dataset/validation'
TEST_DIRECTORY = '../../../aligned dataset/test'
# Labels locations
TRAIN_LABELS_FILE = '../../../train_labels.csv'
VALIDATION_LABELS_FILE = '../../../validation_labels.csv'
TEST_LABELS_FILE = '../../../test_labels.csv'


def make_dirs():
    if not os.path.exists(TRAIN_DIRECTORY):
        os.mkdir(TRAIN_DIRECTORY)
    if not os.path.exists(VALIDATION_DIRECTORY):
        os.mkdir(VALIDATION_DIRECTORY)
    if not os.path.exists(TEST_DIRECTORY):
        os.mkdir(TEST_DIRECTORY)


def group_data_by_labels():
    labels_map = {}
    labels_data = pandas.read_csv(LABELS_FILE, header=None)
    for _, row in labels_data.iterrows():
        image_name = row[0]
        image_label = row[1]
        if image_label in labels_map.keys():
            labels_map[image_label].append(image_name)
        else:
            image_list = [image_name]
            labels_map[image_label] = image_list
    return labels_map


def write_csv(data, filename):
    with open(filename, mode='w') as file:
        for image in data.keys():
            file.write('%s,%s\n' % (image, data[image]))


def read_csv_data(filename):
    labels_map = {}
    data = pandas.read_csv(filename, header=None)
    for _, row in data.iterrows():
        image_name = row[0]
        image_label = row[1]
        labels_map[image_name] = image_label
    return labels_map


def separate_test_and_train_sets():
    labels_map = group_data_by_labels()
    train_set = {}
    validation_set = {}
    test_set = {}
    for label in labels_map.keys():
        images = labels_map[label]
        train_count = 0
        validation_count = 0
        test_count = 0
        for image in images:
            val = random.uniform(0, 1)
            if 0 <= val < 0.8:
                train_set[image] = label
                train_count += 1
            elif 0.8 <= val < 0.9:
                validation_set[image] = label
                validation_count += 1
            else:
                test_set[image] = label
                test_count += 1
        print(label + "\tTrain count: " + str(train_count))
        print(label + "\tValidation count: " + str(validation_count))
        print(label + "\tTest count: " + str(test_count))

    write_csv(train_set, TRAIN_LABELS_FILE)
    write_csv(validation_set, VALIDATION_LABELS_FILE)
    write_csv(test_set, TEST_LABELS_FILE)


def group_images_by_set(train_set, validation_set, test_set):
    for image_name in os.listdir(INPUT_DIRECTORY):
        image_full_name = os.path.join(INPUT_DIRECTORY, image_name)
        target = image_full_name
        if image_name in train_set.keys():
            target = os.path.join(TRAIN_DIRECTORY, image_name)
        elif image_name in validation_set.keys():
            target = os.path.join(VALIDATION_DIRECTORY, image_name)
        elif image_name in test_set.keys():
            target = os.path.join(TEST_DIRECTORY, image_name)
        os.rename(image_full_name, target)


if __name__ == '__main__':
    make_dirs()
    # separate_test_and_train_sets()
    group_images_by_set(read_csv_data(TRAIN_LABELS_FILE),
                        read_csv_data(VALIDATION_LABELS_FILE),
                        read_csv_data(TEST_LABELS_FILE))
