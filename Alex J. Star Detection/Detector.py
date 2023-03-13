import os, tensorflow as tf, matplotlib.pyplot as plt
import random, pandas as pd, cv2, numpy as np, csv
from PIL import Image

PATH = 'C:\\Users\\tomto\\Desktop\\FINAL\\'
DATA_PATH = PATH + 'data\\star\\'
SEP = ';'
DATA = pd.read_csv(PATH + '\\data.csv', sep=SEP)

IMG_H = 64
IMG_W = IMG_H
CHANNELS = 3

LABELS = [6, 8]
DATA_FILENAME = []
DATA_LABEL = []

for id, label in zip(DATA['id'], DATA['rays']):
    DATA_FILENAME.append(id)
    DATA_LABEL.append(label)

TRAIN_SIZE = 0.5

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
        image = cv2.imread(DATA_PATH + rnd_id + ".jpg")

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

        # Get available name
        curr_index = 1
        available = False
        fnames = os.listdir(DATA_PATH)
        while not available:

            ## Iterate over [aug_6rays_1, aug_6rays_2, ...] and get the lateast aug_6rays_<x> to save it
            curr_name = 'aug_6rays_' + str(curr_index)
            available = curr_name + '.jpg' not in fnames
            curr_index += 1

        # Read previous attributes (rays, daidem)
        with open(PATH + csv_name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                id, rays, circle = row[0].split(SEP)
                if rnd_id == id: # Found relevant row
                    break

        # Write new image id with same attributes
        with open(PATH + csv_name, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=SEP,
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([curr_name, rays, circle])

        # Save image in the data\star directory.
        os.chdir(DATA_PATH)
        cv2.imwrite(curr_name + '.jpg', trans_image)

def get_label(file_name):
  # Get label from file name.
  index = DATA_FILENAME.index(file_name)
  label = DATA_LABEL[index]

  # Convert to a true\false vector.
  one_hot = [True if l == label else False for l in LABELS]
    # tf.argmax(one_hot)
  return one_hot

def fname_to_path(fname):
    return DATA_PATH + fname + '.jpg'

def decode_img(img):
  """Returns a tensor from a given image"""
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

def image_to_tensor(file_name):
    img = tf.io.read_file(fname_to_path(file_name))
    img = decode_img(img)
    return img

def process_path(file_name):
    """Returns a tuple of (Tensor, label) from a given filename."""
    label = get_label(file_name)
    return image_to_tensor(file_name), label

def get_data():
    """Returns a tuple of data values (tensor) and labels"""
    x = []
    y = []
    for fname in DATA_FILENAME:
        tens, lab = process_path(file_name=fname)
        x.append(tens)
        y.append(lab)
    return x, y

def get_CNN_model():
    import keras
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
    from keras.losses import categorical_crossentropy

    input_shape = (IMG_H, IMG_W, CHANNELS)
    cnn = Sequential()

    ker = (5, 5)
    cnn.add(Conv2D(8, kernel_size=ker, activation='relu', input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(16, kernel_size=ker, activation='relu', input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(0.25))

    cnn.add(GlobalMaxPooling2D())
    cnn.add(Dense(len(LABELS), activation='softmax'))
    cnn.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    return cnn

def get_pre_trained_model():
    pass

def train_model(model):
    from sklearn.model_selection import train_test_split
    X, y = get_data()
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=2)
    X_train = tf.convert_to_tensor(X_train)
    Y_train = tf.convert_to_tensor(Y_train)
    X_val = tf.convert_to_tensor(X_val)
    Y_val = tf.convert_to_tensor(Y_val)

    return model.fit(x= X_train, y= Y_train,
              shuffle=True,
              batch_size=32,
              epochs=200,
              verbose=2,  ## Print info
              validation_data=(X_val, Y_val))

def plot_history(history):
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

def predict(fname):
    test_img = [image_to_tensor(fname)]
    res = model.predict(tf.convert_to_tensor(test_img))
    ans = res.argmax(axis=-1)[0]
    print(f'Guess : {LABELS[ans]} rays.')

if __name__ == '__main__':
    # balance_data(use_original_only = True, csv_name = 'data.csv')
    model = get_CNN_model()
    history = train_model(model = model)
    plot_history(history)
    print("6:")
    for fname in ['7356426', '7356427', '9065419', '10094593']:
        predict(fname=fname)

    print("8:")
    for fname in ['7903675', '6829257', '4585699']:
        predict(fname=fname)

    # print('?:')
    # for fname in ['8815735', '9674744', 'image00087']:
    #     predict(fname=fname)