import os
import pickle
import timeit

import cv2
import numpy as np
import pandas
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import facial_features_models
from facial_features import detect_facial_features, distances_between_facial_features

VERBOSE = 1
PATIENCE = 50

DATASET_LOCATION = 'dopolnet dataset'

TRAIN_LABELS_FILE = 'train_labels_dopuna.csv'
VALIDATION_LABELS_FILE = 'validation_labels_dopuna.csv'
TEST_LABELS_FILE = 'test_labels_dopuna.csv'

LABELS_MAPPING = {'Fear': 0, 'Disgust': 1, 'Surprise': 2, 'Contempt': 3, 'Anger': 4, 'Neutral': 5, 'Happiness': 6,
                  'Sadness': 7}


def get_labeled_data(filename):
    """
    Creates a two list (train and test), one with facial features data, and the other with labels
    :param filename: the .csv file containing data about image labels
    :return: train set and the labels
    """
    x_data = []
    y_data = []
    data_frame = pandas.read_csv(filename, header=None)
    for i, row in data_frame.iterrows():
        img_name = row[0]
        img_label = row[1]
        if i % 1000 == 0:
            print('processing %d...' % i)
            print(img_name)
        image_path = os.path.join(DATASET_LOCATION, img_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            facial_features = detect_facial_features(image, True)
            distances = distances_between_facial_features(facial_features)
            x_data.append(distances)
            y_data.append(LABELS_MAPPING[img_label])
        except:
            print('Error reading features for: %s' % img_name)
    return x_data, y_data


def train_and_test_nn(x_train, y_train, x_validation, y_validation, x_test, y_test):
    """
    Performs training and testing of a Neural network.
    :param x_train: train data
    :param y_train: train labels
    :param x_validation: validation data
    :param y_validation: validation labels
    :param x_test: test data
    :param y_test: test labels
    """
    x_train = np.array([np.atleast_2d(x) for x in x_train])
    y_train = to_categorical(np.array(y_train), 8)

    x_validation = np.array([np.atleast_2d(x) for x in x_validation])
    y_validation = to_categorical(np.array(y_validation), 8)

    x_test = np.array([np.atleast_2d(x) for x in x_test])
    y_test = to_categorical(np.array(y_test), 8)

    # Model
    print('Model')
    print(x_train[0].shape)
    model = facial_features_models.big_XCEPTION(x_train[0].shape, 8)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # define logging
    log_file_path = 'emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)

    # define early stopping and reducing learning rate on validation loss plateau
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE / 4), verbose=1)

    callbacks = [csv_logger, early_stop, reduce_lr]

    # Train
    batch_size = 32
    num_epochs = 300
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=num_epochs, verbose=VERBOSE,
              callbacks=callbacks, validation_data=(x_validation, y_validation))

    # Test
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Model score: {}".format(score))

    # Save model
    model_filename = 'big_X[dropout_single_0.5][%d epochs][%d batch].h5' % (num_epochs, batch_size)
    model.save_weights(model_filename)


def train_and_test_with_classifier(cls_name, classifier, train_features, train_labels, test_features, test_labels):
    """
    Performs training and testing of the provided classifier
    :param cls_name: name of the classifier
    :param classifier: the classifier
    :param train_features: train data
    :param train_labels: train labels
    :param test_features: test data
    :param test_labels: test features
    """
    # Training
    print('Training: %s on %d features. Train set size: %d' % (cls_name, len(train_features[0]), len(train_features)))
    start_time = timeit.default_timer()
    classifier.fit(train_features, train_labels)
    print('Training finished in: %.2f' % (timeit.default_timer() - start_time))

    # Testing
    print('Testing: %s...' % cls_name)
    start_time = timeit.default_timer()
    result = classifier.predict(test_features)
    print('Testing finished in: %.2f' % (timeit.default_timer() - start_time))

    accuracy = 0
    for i in range(0, len(result)):
        if result[i] == test_labels[i]:
            accuracy += 1
    print('Accuracy: %.3f' % (float(accuracy) / float(len(result))))
    print('Correct guesses: %d/%d\n' % (accuracy, len(result)))

    # Pickle the results
    results_pickle_fname = 'results_' + cls_name + '.pickle'
    with open(results_pickle_fname, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    x_train_pickle = 'x_train.pickle'
    y_train_pickle = 'y_train.pickle'
    x_validation_pickle = 'x_validation.pickle'
    y_validation_pickle = 'y_validation.pickle'
    x_test_pickle = 'x_test.pickle'
    y_test_pickle = 'y_test.pickle'

    # TRAIN
    if os.path.exists(x_train_pickle):
        with open(x_train_pickle, mode='rb') as f:
            x_train = pickle.load(f)
        with open(y_train_pickle, mode='rb') as f:
            y_train = pickle.load(f)
    else:
        x_train, y_train = get_labeled_data(TRAIN_LABELS_FILE)
        with open(x_train_pickle, mode='wb') as f:
            pickle.dump(x_train, f)
        with open(y_train_pickle, mode='wb') as f:
            pickle.dump(y_train, f)

    # VALIDATION
    if os.path.exists(x_validation_pickle):
        with open(x_validation_pickle, mode='rb') as f:
            x_validation = pickle.load(f)
        with open(y_validation_pickle, mode='rb') as f:
            y_validation = pickle.load(f)
    else:
        x_validation, y_validation = get_labeled_data(VALIDATION_LABELS_FILE)
        with open(x_validation_pickle, mode='wb') as f:
            pickle.dump(x_validation, f)
        with open(y_validation_pickle, mode='wb') as f:
            pickle.dump(y_validation, f)

    # TEST
    if os.path.exists(x_test_pickle):
        with open(x_test_pickle, mode='rb') as f:
            x_test = pickle.load(f)
        with open(y_test_pickle, mode='rb') as f:
            y_test = pickle.load(f)
    else:
        x_test, y_test = get_labeled_data(TEST_LABELS_FILE)
        with open(x_test_pickle, mode='wb') as f:
            pickle.dump(x_test, f)
        with open(y_test_pickle, mode='wb') as f:
            pickle.dump(y_test, f)

    classifiers = {
        'random_forest_100': RandomForestClassifier(n_estimators=100, random_state=0),
        'random_forest_250': RandomForestClassifier(n_estimators=250, random_state=0),
        'random_forest_500': RandomForestClassifier(n_estimators=500, random_state=0),
        'extra_trees_100': ExtraTreesClassifier(n_estimators=100),
        'extra_trees_250': ExtraTreesClassifier(n_estimators=250),
        'extra_trees_500': ExtraTreesClassifier(n_estimators=500),
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression()
    }

    print('=========Training and testing NN.==========')
    train_and_test_nn(x_train, y_train, x_validation, y_validation, x_test, y_test)

    print('=========Training other classifiers.==========')
    for clf_name in classifiers.keys():
        train_and_test_with_classifier(clf_name, classifiers[clf_name],
                                       train_features=x_train + x_validation,
                                       train_labels=y_train + y_validation,
                                       test_features=x_test,
                                       test_labels=y_test)
    print('=========Training finished.==========')
