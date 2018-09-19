from keras import Input, layers, Model
from keras.layers import BatchNormalization, Activation, Dropout, Conv1D, \
    SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D


def big_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv1D(32, (1), strides=(2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv1D(64, (1), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv1D(128, (1), strides=(2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv1D(128, (1), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv1_act')(x)
    x = SeparableConv1D(128, (1), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling1D((3), strides=(2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv1D(256, (1), strides=(2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv1D(256, (1), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = Dropout(0.5)(x)
    x = SeparableConv1D(256, (1), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling1D((3), strides=(2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv1D(num_classes, (1),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model
