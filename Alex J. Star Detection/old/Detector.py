import os, tensorflow as tf, matplotlib.pyplot as plt
import random, pandas as pd, cv2, numpy as np, csv
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import keras
import keras.utils
from keras.utils import to_categorical
from keras import regularizers

from keras import layers
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from keras.layers import Input, GlobalMaxPooling2D, Dropout, Dense, Flatten

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.models import Sequential
from sklearn.model_selection import train_test_split

PATH = 'C:\\Users\\tomto\\Desktop\\FINAL\\'
DATA_PATH = PATH + 'data\\star_side\\'
SEP = ';'
DATA = pd.read_csv(PATH + '\\data.csv', sep=SEP)

IMG_H = 64
IMG_W = IMG_H
CHANNELS = 3

LABELS = [6, 8]
DATA_FILENAME = []
DATA_LABEL = []

for id, label in zip(DATA['id'], DATA['rays']):
    if label != 0:
        DATA_FILENAME.append(id)
        DATA_LABEL.append(label)

TRAIN_SIZE = 0.5

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def balance_data(use_original_only:bool, csv_name:str):
    """Clones transformed data, there is an even number of samples with different labels.
    Appends cloned data information to data.csv"""
    counter = DATA['rays'].value_counts()
    label_counter = {}

    # For each relevant label, count appearances.
    for lbl in counter.keys():
        if lbl in LABELS:
            label_counter[lbl] = counter[lbl]

    most_common = max(label_counter)
    least_common = min(label_counter)

    # Select group to clone
    kernel = []
    for i, label in enumerate(DATA_LABEL):
        fname = DATA_FILENAME[i]
        relevant = label == least_common and not (use_original_only and 'aug' in str(fname))
        if relevant: kernel.append(fname)

    # For each image, randomly rotate and move around, then save it with the same csv information
    # Do it until there is the same amount of pictures for both of the classes.
    for k in range(label_counter[most_common] - label_counter[least_common]):
        rnd_id = str(random.sample(kernel, 1)[0]) # Get random sample to clone
        label = get_label(rnd_id)
        label = str(label)

        curr_dir = DATA_PATH + label + "\\"
        image = cv2.imread(curr_dir + rnd_id + ".jpg")
        image = change_brightness(image, value= 40 - random.randint(0, 80))  # increases

        # dividing height and width by 2 to get the center of the image
        height, width = image.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width / 2, height / 2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix, with a random angle
        theta = random.randint(0, 360)
        rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = theta, scale = 1)

        # Randomly move image around
        w = int(width / 8)
        h = int(height / 8)
        a = random.randint(- w, w)
        b = random.randint(- h, h)
        trans_matrix = np.float32([[1, 0, a], [0 ,1, b]])

        # transform the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src = image, M = rotate_matrix, dsize = (width, height))
        trans_image = cv2.warpAffine(src = rotated_image, M = trans_matrix, dsize = (width, height))

        # Show image while changing it.
        # plt.imshow(trans_image)
        # plt.show()

        # Get available name
        curr_index = 1
        available = False
        fnames = os.listdir(curr_dir)
        while not available:

            ## Iterate over [aug_6rays_1, aug_6rays_2, ...] and get the lateast aug_6rays_<x> to save it
            curr_name = 'aug_6rays_' + str(curr_index)
            available = curr_name + '.jpg' not in fnames
            curr_index += 1

        ## Write in data.csv the new clone, not needed if data is loaded directly from directory. ##
        # # Read previous attributes (rays, daidem)
        # with open(PATH + csv_name, newline='') as csvfile:
        #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #     for row in spamreader:
        #         id, rays, circle = row[0].split(SEP)
        #         if rnd_id == id: # Found relevant row
        #             break
        #
        # # Write new image id with same attributes
        # with open(PATH + csv_name, 'a', newline='') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=SEP,
        #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow([curr_name, rays, circle])

        # Save image in the data\star directory.
        os.chdir(curr_dir)
        cv2.imwrite(curr_name + '.jpg', trans_image)

def get_label(file_name):
    """Returns the label (6 or 8) of a sample"""
    index = DATA_FILENAME.index(file_name)
    return DATA_LABEL[index]

def get_label_vec(file_name):
  # Get label from file name.
  index = DATA_FILENAME.index(file_name)
  label = DATA_LABEL[index]

  # Convert to a true\false vector.
  one_hot = [True if l == label else False for l in LABELS]
    # tf.argmax(one_hot)
  return one_hot

def fname_to_path(fname):
    """Returns the same filename but with a prefix of the path, and a postfix .jpg"""
    index = DATA_FILENAME.index(fname)
    label = DATA_LABEL[index]
    suffix = fname + '.jpg' if not fname.endswith('.jpg') else fname
    return f'{DATA_PATH}{label}\\{suffix}'

def decode_img(img):
  """Returns a tensor from a given, resized image"""
  img = tf.io.decode_jpeg(img, channels = CHANNELS)

  # plt.imshow(img)
  # plt.show()

  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, size = [IMG_H, IMG_W])

  # plt.imshow(img)
  # plt.show()

  return img

def tensor_to_image(tensor):
    """Returns an image from a given tensor"""
    tensor = tensor*255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def image_to_tensor(file_name, star=True):
    """Attempts to find image, and then returns a tensor from the resized image."""
    path = fname_to_path(file_name)
    if not star:
        path = file_name
    try:
        img = tf.io.read_file(path)
    except Exception as e:
        print(e)
        print(f'{file_name} not found.')
        return None
    img = decode_img(img)
    return img

def process_path(file_name):
    """(Unused) Returns a tuple of (Tensor, label) from a given filename."""
    label = get_label_vec(file_name)
    return image_to_tensor(file_name), label

def get_data():
    """[Unused anymore, from manual loading swithced to keras loading from directory]
    Returns a tuple of data values (tensor) and labels"""
    x = []
    y = []
    for fname in DATA_FILENAME:
        temp = process_path(file_name=fname)
        tens = temp[0]
        lab = temp[1]
        x.append(tens)
        y.append(lab)
    return x, y

def get_CNN_model():
    """Returns a CNN model using tf2 and keras."""
    z = 0.01
    reg = None # regularizers.l2(0.0001)

    cnn = tf.keras.Sequential([

        # layers.RandomZoom(z),
        # layers.RandomContrast(z),
        # layers.RandomBrightness(factor=z),
        # layers.RandomRotation(factor=(-z, z)),

        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),

        layers.Dense(64, activation='relu', kernel_regularizer = reg),
        Dropout(0.25),
        Dense(len(LABELS), activation='softmax')
    ])

    cnn.compile(optimizer='rmsprop',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return cnn

"""
https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4

"""
def vgg_model():
    vgg19 = VGG19(weights='imagenet', include_top=False,
                  input_shape=(IMG_H, IMG_W, CHANNELS), classes=len(LABELS))
    for layer in vgg19.layers:
        layer.trainable = False

    input = Input(shape=(IMG_H, IMG_W, CHANNELS), name='input')
    x = vgg19(input, training=False)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(LABELS), activation = 'softmax')(x)

    model = keras.models.Model(input, outputs)

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy'])

    return model

def train_model(model):
    train, validate = keras.utils.image_dataset_from_directory(DATA_PATH,
                                                               shuffle=False,
                                                               image_size = (IMG_H, IMG_W),
                                                               validation_split = 1-TRAIN_SIZE,
                                                               subset='both',
                                                               seed = 2)

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    return model.fit(train,
              shuffle=True,
              batch_size= 32,
              epochs=100,
              verbose=2,  ## Print info
              validation_data=validate,
                  callbacks=[callback])

def plot_history(history):
    """Plots two graphs, one for accuracy, and one for loss. Both depicting training and validation information."""
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(history.history['val_accuracy'], label='validation')  ## Val
    ax1.plot(history.history['accuracy'], label='train')  ## Train

    ax2.plot(history.history['val_loss'], label='validation')
    ax2.plot(history.history['loss'], label='train')

    ax1.set_xlabel('epochs')
    ax1.set_title('accuracy')
    ax1.legend()
    ax2.set_xlabel('epochs')
    ax2.set_title('loss')
    ax2.legend()

    plt.show()
    plt.show()

def predict(model, fname, star=True):
    test_img = [image_to_tensor(fname, star)]
    res = model.predict(tf.convert_to_tensor(test_img), verbose=0)
    ans = res.argmax(axis=-1)[0]
    print(res)
    print(f'Guess : {LABELS[ans]} rays.')

def confusion_matrix_clf(model):
    x, y = get_data()

    predictions = model.predict(tf.convert_to_tensor(x), verbose=0)
    predictions = [LABELS[i] for i in predictions.argmax(axis=-1)]
    real = [LABELS[i.index(True)] for i in y]

    wrong_fnames = []
    for i in range(len(predictions)):
        if predictions[i] != real[i]:
            wrong_fnames.append(DATA_FILENAME[i])

    cm = confusion_matrix(real, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.show()

    return wrong_fnames

def show_images(img_list):
    for img in img_list:
        curr_dir = DATA_PATH + str(get_label(img)) + "\\"
        plt.imshow(cv2.imread(curr_dir + img + ".jpg"))
        plt.title("File " + img + " , Label:" + str(get_label(img)))
        plt.show()

if __name__ == '__main__':
    # balance_data(use_original_only = True, csv_name = 'data.csv')
    model = get_CNN_model()
    history = train_model(model)
    plot_history(history)

    failures = confusion_matrix_clf(model)
    show_images(failures)

    #
    # print("6:")
    # for fname in ['7356426', '7356427', '9065419', '1342054']:
    #     predict(model, fname=fname)
    #
    # print("8:")
    # for fname in ['3976933', '3978221', '3960343', '8488211']:
    #     predict(model, fname=fname)

    # print("?:")
    # for fname in os.listdir(PATH + 'data\\unsure\\'):
    #     predict(model, fname=PATH + 'data\\unsure\\' + fname, star=False)
