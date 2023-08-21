"""
Model Library, usages:
* Training a given model with class weights
* Defining a fully connected model.
* Defining a CNN model. (This is what is mainly used)
* Defining a vgg \ mobile_net model. (Pretrained model)
"""

from Cons import *


def train_model(model, w):
    """
    :param model: A deep learning model
    :param w: A set of weights
    :return: Returns the history of the model after training for a a set amount of epochs set in CONS
    """
    train, validate = keras.utils.image_dataset_from_directory(DATA_PATH,
                                                               shuffle = False,
                                                               image_size = (IMG_H, IMG_W),
                                                               validation_split = 1 - TRAIN_SIZE,
                                                               subset = 'both',
                                                               seed = 2)

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    return model.fit(train,
              shuffle=True,
              batch_size= BATCH_SIZE,
              epochs=EPOCHS,
              verbose=0,  ## Print info
              validation_data=validate,
              class_weight = w,
              callbacks = [callback])


def mobile_net():
    """

    :return: Pretrained model with all frozen layers but the last two.
    """
    IMG_H = 224
    IMG_W = IMG_H
    # https://keras.io/api/applications/#usage-examples-for-image-classification-models
    # MobileNet was chosen to be tested due to small depth and small number of parameters.
    from keras.applications.mobilenet import MobileNet
    input_layer = Input((IMG_H, IMG_W, CHANNELS), name='input_layer')
    incep_res = MobileNet(include_top=False, weights='imagenet', input_tensor=input_layer)
    for layer in incep_res.layers:
        layer.trainable = False

    # pooling to reduce dimensionality of each feature map
    x = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding='same')(incep_res.output)
    x = Flatten(name='Flatten1')(x)
    x = BatchNormalization(name='BatchNormFlat')(x)

    x = Dense(128, activation='relu', activity_regularizer='l2')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(32, activation='relu', activity_regularizer='l2')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(len(LABELS))(x)
    output_layer = Activation('softmax', name='Softmax')(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    return model


def get_FN_model():
    """

    :return: A fully connected model (Dense layers connected to each other)
    """
    z = 0.01
    reg = None # regularizers.l2(0.0001)

    fc = tf.keras.models.Sequential([
        layers.RandomZoom(z),
        layers.RandomContrast(z),
        layers.RandomBrightness(factor=z),
        layers.RandomRotation(factor=(-z, z)),

        Flatten(input_shape=(IMG_H, IMG_W, CHANNELS)),
        Dense(1024, activation='relu', kernel_regularizer = reg),
        Dropout(0.5),
        Dense(len(LABELS), activation='softmax')
    ])

    fc.compile(optimizer='rmsprop',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return fc


def get_CNN_model(conv_num, fil_num, fil_size, opt, activ):
    """

    :param conv_num: Amount of convolutions before the final softmax activation.
    :param fil_num: For each convolution, sets the amount of filters.
    :param fil_size: For all convolutions layers, set this as the filter size. (should be around 3 - 9)
    :param opt: Sets the optimizer for updating the weights
    :param activ: Activation function of each layer.
    :return: The defined model
    """
    z = random.random() * 0.01
    z = round(z, 4)
    reg_str = random.random() * 0.001
    reg_str = round(reg_str, 4)

    print(f'Random augmentation strength - {z}')
    print(f'Regulizer strength - {reg_str}')
    reg = regularizers.l2(reg_str)

    cnn = tf.keras.Sequential()
    cnn.add(layers.RandomRotation(z))
    cnn.add(layers.RandomZoom(z))
    cnn.add(layers.GaussianNoise(0.2))

    layer = 1
    for _ in range(conv_num):
        cnn.add(Conv2D(fil_num, fil_size, kernel_regularizer = reg, activation = activ))
        cnn.add(MaxPooling2D())
        layer *= 2

    cnn.add(Flatten())
    cnn.add(Dense(len(LABELS), activation='softmax'))

    cnn.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return cnn


"""
https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4

"""
def get_vgg_model():
    """

    :return: Pretrained model with all but the last Dense layer frozen.
    """
    from keras.applications import VGG19
    vgg19 = VGG19(weights='imagenet', include_top=False,
                  input_shape=(IMG_H, IMG_W, CHANNELS), classes=len(LABELS))
    for layer in vgg19.layers:
        layer.trainable = False

    input = Input(shape=(IMG_H, IMG_W, CHANNELS), name='input')

    x = vgg19(input, training=False)
    x = Flatten()(x)

    for _ in range(1):
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)

    outputs = Dense(len(LABELS), activation = 'softmax')(x)

    model = keras.models.Model(input, outputs)

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    return model
