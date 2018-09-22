import os

from training.dataset import read_data
from training.preprocess_images import preprocess_images

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from training.models import mini_XCEPTION

# number of classes inside the dataset
NUMBER_OF_CLASSES = 8
# number of epochs to train the neural network on
EPOCHS = 3
# number of batch size
BATCH_SIZE = 128
# the shape of the images which will be used for the training
INPUT_SHAPE = (64, 64, 1)

VERBOSE = 1
PATIENCE = 50
TRAINED_MODELS_DIRECTORY = 'trained_models/'
LABELS_DIRECTORY = 'labels/'


def train_nn(images_directory):
    # first thing we do is create the path to the saved model
    trained_models_path = TRAINED_MODELS_DIRECTORY + "mini_XCEPTION_{}-epochs_{}-batch_size.h5".\
        format(EPOCHS, BATCH_SIZE)

    if os.path.exists(trained_models_path):
        print("Saved model with number of epochs: {} and batch_size: {} saved at: {} already exists. Training canceled.".
              format(EPOCHS, BATCH_SIZE, trained_models_path))
        return

    model = mini_XCEPTION(input_shape=INPUT_SHAPE, num_classes=NUMBER_OF_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()

    # define logging
    log_file_path = 'emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)

    # define early stopping and reducing learning rate on validation loss plateau
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/4), verbose=1)

    callbacks = [csv_logger, early_stop, reduce_lr]

    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)

    x_train, y_train = read_data(os.path.join(LABELS_DIRECTORY, "combined_train_labels.csv"), images_directory, NUMBER_OF_CLASSES)

    x_val, y_val = read_data(os.path.join(LABELS_DIRECTORY, "validation_labels.csv"), images_directory, NUMBER_OF_CLASSES)

    data_generator.fit(x_train)

    model.fit_generator(data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
                        callbacks=callbacks, validation_data=(x_val,y_val))

    x_test, y_test = read_data(os.path.join(LABELS_DIRECTORY,
                                            "test_labels.csv"), images_directory, NUMBER_OF_CLASSES)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print("Model score: {}".format(score))

    model.save_weights(trained_models_path)


if __name__ == '__main__':
    source_images_location = 'combined_images/'

    images_location = preprocess_images(source_images_location, INPUT_SHAPE)

    if not os.path.exists(TRAINED_MODELS_DIRECTORY):
        os.makedirs(TRAINED_MODELS_DIRECTORY)

    train_nn(images_directory=images_location)