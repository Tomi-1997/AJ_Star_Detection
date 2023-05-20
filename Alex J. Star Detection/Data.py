import random

from Cons import *


def get_label(file_name):
    """Returns the label (0, 6 or 8) of a sample"""
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


def fname_to_path(fname, test):
    """Returns the same filename but with a prefix of the path, and a postfix .jpg"""
    path = TEST_PATH if test else DATA_PATH
    index = DATA_FILENAME.index(fname)
    label = DATA_LABEL[index]
    suffix = fname + '.jpg' if not fname.endswith('.jpg') else fname
    return f'{path}{label}\\{suffix}'


def decode_img(img):
    """Returns a tensor from a given, resized image"""
    img = tf.io.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[IMG_H, IMG_W])
    return img


def tensor_to_image(tensor):
    """Returns an image from a given tensor"""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def image_to_tensor(file_name, test=False):
    """Attempts to find image, and then returns a tensor from the resized image."""
    path = fname_to_path(file_name, test)
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img


def process_path(file_name, test=False):
    """Returns a tuple of (Tensor, label) from a given filename."""
    label = get_label_vec(file_name)
    return image_to_tensor(file_name, test), label


def get_data(test=False):
    """test - False
    Returns a tuple of all data values (tensor) and labels from data folder
    test - True
    Returns a tuple of all data values (tensor) and labels from test folder"""
    x = []
    y = []

    if len(TEST) == 0:  # Test list is not initialized
        f0 = os.listdir(DATA_PATH + "\\0")
        f6 = os.listdir(DATA_PATH + "\\6")
        f8 = os.listdir(DATA_PATH + "\\8")
        train_files = f0 + f6 + f8
        for fname in DATA_FILENAME:
            if fname + '.jpg' not in train_files:
                TEST.append(fname)

    src = TEST if test else DATA_FILENAME
    for fname in src:

        if fname in TEST and not test: continue  # Don't test images from the test folder / list
        temp = process_path(file_name=fname, test=test)
        tens = temp[0]
        lab = temp[1]

        lab_ind = lab.index(True)
        lab_str = LABELS[lab_ind]

        x.append(tens)
        y.append(lab)

    return x, y


def update_csv_by_tag(tag):
    """
updates csv per tag (0 / 6 / 8) for any additional data
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
    if "IteratorGetNext" in input.name:
        return input
    sig = 0
    ker = 9
    smoothed = cv2.GaussianBlur(input, (ker, ker), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_DEFAULT)
    cny = cv2.Canny(smoothed, 10, 150)
    return cny


def draw_edges(img):
    cny = find_edges(img)
    image_copy = Image.copy(img)
    contours, hierarchy = cv2.findContours(cny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1)


def move_train_to_dir(num : int, classes : list):
    """Transfer X images from train directory to test directory, for each given class"""
    for label in classes:

        pop = os.listdir(DATA_PATH + str(label))

        num = min(len(pop), num)
        target = random.sample(pop, num)

        for filename in target:
            old_file = DATA_PATH + str(label) + "\\" + filename
            destination = TEST_PATH + str(label) + "\\" + filename

            os.rename(old_file, destination)


def move_all_back(classes : list):
    """Transfers all images from test directory to train directory for each given class."""
    for label in classes:

        target = os.listdir(TEST_PATH + str(label))
        old_path = TEST_PATH + str(label) + "\\"
        new_path = DATA_PATH + str(label) + "\\"

        for filename in target:

            if not filename.endswith('.jpg'): continue

            old_file =  old_path + filename
            destination = new_path + filename

            os.rename(old_file, destination)
# if __name__ == '__main__':
#     DATA = pd.read_csv(os.getcwd() + '\\data.csv', sep=SEP)
#     for did, label in zip(DATA['id'], DATA['rays']):
#         DATA_FILENAME.append(did)
#         DATA_LABEL.append(label)
#
#     update_csv_by_tag("8")
