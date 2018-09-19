# Full data-set
import random

import pandas

INPUT_DIRECTORY = '../../../dopolnet dataset'
LABELS_FILE_1 = '../../../updated_labels.csv'
LABELS_FILE_2 = '../../../labels2.csv'

# Labels locations
TRAIN_LABELS_FILE = '../../../train_labels_dopuna.csv'
VALIDATION_LABELS_FILE = '../../../validation_labels_dopuna.csv'
TEST_LABELS_FILE = '../../../test_labels_dopuna.csv'

LABELS = {}


def group_data_by_labels():
    """
    Groups the images by their label.
    The format is dictionary with key=label, and value=list of all image names with that label.
    :return: the dictionary
    """
    labels_to_images = {}

    for img_name in LABELS.keys():
        img_label = LABELS[img_name]
        if img_label in labels_to_images.keys():
            labels_to_images[img_label].append(img_name)
        else:
            img_list = [img_name]
            labels_to_images[img_label] = img_list
    return labels_to_images


def separate_train_validation_test(train_ratio, validation_ratio, test_ratio):
    """
    Separates the full data set into 3 sets (train, validation and test) according to the provided ratios.
    The distribution is random.
    The three sets are then written into 3 .csv files with format: (image_name, label)
    :param train_ratio: ratio for the train set
    :param validation_ratio: ratio for the validation set
    :param test_ratio: ratio for the test set
    :return:
    """
    train_range = 0 + train_ratio
    validation_range = train_range + validation_ratio
    test_range = validation_range + test_ratio

    labels_to_images = group_data_by_labels()

    train_set = {}
    validation_set = {}
    test_set = {}

    for label in labels_to_images.keys():
        img_list = labels_to_images[label]
        train_count = 0
        validation_count = 0
        test_count = 0
        for image in img_list:
            val = random.uniform(0, 1)
            if 0 <= val < train_range:
                train_set[image] = label
                train_count += 1
            elif train_range <= val < validation_range:
                validation_set[image] = label
                validation_count += 1
            else:
                test_set[image] = label
                test_count += 1
        print("%s\tTrain count: %d" % (label, train_count))
        print("%s\tValidation count: %d" % (label, validation_count))
        print("%s\tTest count: %d" % (label, test_count))

    write_csv(train_set, TRAIN_LABELS_FILE)
    write_csv(validation_set, VALIDATION_LABELS_FILE)
    write_csv(test_set, TEST_LABELS_FILE)


def write_csv(data, filename):
    with open(filename, mode='w') as file:
        for image in data.keys():
            file.write('%s,%s\n' % (image, data[image]))


if __name__ == '__main__':
    data_frame_1 = pandas.read_csv(LABELS_FILE_1, header=None)
    for _, row in data_frame_1.iterrows():
        img_name = row[0]
        img_label = row[1]
        LABELS[img_name] = img_label

    data_frame_2 = pandas.read_csv(LABELS_FILE_2, header=None)
    for _, row in data_frame_2.iterrows():
        img_name = row[0]
        img_label = row[1]
        LABELS[img_name] = img_label

    separate_train_validation_test(0.70, 0.15, 0.15)
