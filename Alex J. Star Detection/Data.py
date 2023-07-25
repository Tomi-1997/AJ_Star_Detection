"""
Data Library, usages:
* Filepath to tensor.
* Get batch of data for testing.
* CSV Updates.
* Data files transferring. (To and from train or test folders)

"""

import os
import random
from Cons import *


def decode_img(img):
    """Returns a resized tensor (from an image) according to the height and width defined in CONS"""
    img = tf.io.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[IMG_H, IMG_W])
    return img


def to_tensor(filepath, label):
    """
    :param filepath F:
    :param label L:
    :return: A tuple (x, y) x : Resized tensor from F, y : one hot vector of L
    """
    one_hot = [True if l == label else False for l in LABELS]
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img, one_hot


def get_data(test = False, TEST = None):
    """test - False
    Returns a tuple of all data values (tensor) and labels from data folder
    test - True
    Returns a tuple of all data values (tensor) and labels from test folder"""
    x = []
    y = []

    path = TEST_PATH if test else DATA_PATH
    #
    # if len(TEST) == 0:  # Test list is not initialized
    #     f0 = os.listdir(DATA_PATH + "\\0")
    #     f6 = os.listdir(DATA_PATH + "\\6")
    #     f8 = os.listdir(DATA_PATH + "\\8")
    #     train_files = f0 + f6 + f8
    #     for fname in DATA_FILENAME:
    #         if fname + '.jpg' not in train_files:
    #             TEST.append(fname)

    src = TEST_PATH if test else DATA_FILENAME
    for label in os.listdir(src):
        for filename in os.listdir(src + "\\" + label):
            if filename.endswith(".ini"): continue

            temp = to_tensor(src + "\\" + label + "\\" + filename, int(label))
            tens = temp[0]
            lab = temp[1]

            x.append(tens)
            y.append(lab)

            if TEST is not None: TEST.append(filename)

    return x, y


def update_csv_by_tag(tag):
    """
    For each unwritten sample in the CSV file, adds it along with it's label.
    """
    assert (tag == 6 or tag == 0 or tag == 8)
    f_data = []
    f_dir = os.listdir(DATA_PATH + "\\" + tag)
    for filename in f_dir:
        if filename.endswith(".jpg"):
            f_data.append(filename.split(".")[0])

    for name in f_data:
        if name not in DATA_FILENAME:
            with open("data.csv", "a") as infile:
                # Create a writer object for csv
                writer = csv.writer(infile, delimiter=';')
                # Data we want to write to the CSV file
                line = [name, tag, "-1"]
                # Write the row the CSV file.
                # Note that this line updates the file in-place
                writer.writerow(line)


def find_edges(input):
    """A layer to find edges within an image, (*) currently unused, as CNN layers automatically do it. """
    if "IteratorGetNext" in input.name:
        return input
    sig = 0
    ker = 9
    smoothed = cv2.GaussianBlur(input, (ker, ker), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_DEFAULT)
    cny = cv2.Canny(smoothed, 10, 150)
    return cny


def draw_edges(img):
    """Draws the edges found in the find_edges() function. (*) currently unused. """
    cny = find_edges(img)
    image_copy = Image.copy(img)
    contours, hierarchy = cv2.findContours(cny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1)


def move_train_to_dir(num : int, classes : list):
    """
    Transfer X images from train directory to test directory, for each given class.
    This is done each iteration in order to test which models can predict successfully an unseen image.
    """
    for label in classes:

        pop = os.listdir(DATA_PATH + str(label))
        pop = list(filter(lambda x: x.endswith(".jpg"), pop))
        num = min(len(pop), num)
        target = random.sample(pop, num)

        for filename in target:
            old_file = DATA_PATH + str(label) + "\\" + filename
            destination = TEST_PATH + str(label) + "\\" + filename

            os.rename(old_file, destination)


def move_all_back(classes : list):
    """
    Transfers ALL images from test directory to train directory for each given class.
    It is important to shuffle what images are defined as train and what are test for each iteration,
    to diversify the saved models. (Models are saved only if they correctly guess unseen images)
    """
    for label in classes:

        target = os.listdir(TEST_PATH + str(label))
        old_path = TEST_PATH + str(label) + "\\"
        new_path = DATA_PATH + str(label) + "\\"

        for filename in target:

            if not filename.endswith('.jpg'): continue

            old_file =  old_path + filename
            destination = new_path + filename

            os.rename(old_file, destination)
