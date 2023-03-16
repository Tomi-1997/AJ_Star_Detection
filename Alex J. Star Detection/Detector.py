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
    return DATA_PATH + fname + '.jpg' if not fname.endswith('.jpg') else DATA_PATH + fname

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

def image_to_tensor(file_name, star):
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
        tens, lab = process_path(file_name=fname)
        x.append(tens)
        y.append(lab)
    return x, y

def get_CNN_model():
    """Returns a CNN model using tf2 and keras."""
    import keras
    from tensorflow.keras import layers
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D

    z = 0.05
    cnn = tf.keras.Sequential([
        layers.RandomZoom(z),
        layers.RandomContrast(z),
        layers.RandomBrightness(factor=z),

        Conv2D(8, 3, padding='same', activation='relu'), MaxPooling2D(),
        Conv2D(16, 3, padding='same', activation='relu'), MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(),

        Flatten(),

        layers.Dense(128, activation='relu'), Dropout(0.25),
        Dense(len(LABELS), activation='softmax')
    ])

    cnn.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return cnn

def get_pre_trained_model():
    pass

def train_model(model):
    from sklearn.model_selection import train_test_split
    import keras.utils

    train, validate = keras.utils.image_dataset_from_directory(DATA_PATH,
                                                               image_size = (IMG_H, IMG_W),
                                                               validation_split = 1-TRAIN_SIZE,
                                                               subset='both',
                                                               seed = 2)
    return model.fit(train,
              shuffle=True,
              batch_size=16,
              epochs=50,
              verbose=2,  ## Print info
              validation_data=validate)

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

def predict(fname, star=True):
    test_img = [image_to_tensor(fname, star)]
    res = model.predict(tf.convert_to_tensor(test_img), verbose=0)
    ans = res.argmax(axis=-1)[0]
    print(res)
    print(f'Guess : {LABELS[ans]} rays.')

if __name__ == '__main__':
    # balance_data(use_original_only = True, csv_name = 'data.csv')
    model = get_CNN_model()
    history = train_model(model = model)
    plot_history(history)

    print("6:")
    for fname in ['7356426', '7356427', '9065419', '1342054']:
        predict(fname='6\\'+fname)

    print("8:")
    for fname in ['3976933', '3978221', '3960343', '8488211']:
        predict(fname='8\\'+fname)

    print("?:")
    for fname in os.listdir(PATH + 'data\\unsure\\'):
        predict(fname=PATH + 'data\\unsure\\' + fname, star=False)
