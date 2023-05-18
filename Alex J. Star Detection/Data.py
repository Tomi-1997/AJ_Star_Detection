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


def image_to_tensor(file_name, test = False):
    """Attempts to find image, and then returns a tensor from the resized image."""
    path = fname_to_path(file_name, test)
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img


def process_path(file_name, test = False):
    """Returns a tuple of (Tensor, label) from a given filename."""
    label = get_label_vec(file_name)
    return image_to_tensor(file_name, test), label


def get_data(test = False):
    """test - False
    Returns a tuple of all data values (tensor) and labels from data folder
    test - True
    Returns a tuple of all data values (tensor) and labels from test folder"""
    x = []
    y = []

    if len(TEST) == 0: ## Test list is not initialized
        f0 = os.listdir(DATA_PATH + "\\0")
        f6 = os.listdir(DATA_PATH + "\\6")
        f8 = os.listdir(DATA_PATH + "\\8")
        train_files = f0 + f6 + f8
        for fname in DATA_FILENAME:
            if fname + '.jpg' not in train_files:
                TEST.append(fname)


    src = TEST if test else DATA_FILENAME
    for fname in src:

        if fname in TEST and not test: continue ## Don't test images from the test folder / list
        temp = process_path(file_name=fname, test=test)
        tens = temp[0]
        lab = temp[1]

        lab_ind = lab.index(True)
        lab_str = LABELS[lab_ind]

        x.append(tens)
        y.append(lab)

    return x, y


def find_edges(input):
    if "IteratorGetNext" in input.name:
        return input
    sig = 0
    ker = 9
    smoothed = cv2.GaussianBlur(input, (ker, ker), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_DEFAULT)
    cny = cv2.Canny(smoothed, 10, 150)
    return cny


def draw_edges(img):
    contours, hierarchy = cv2.findContours(cny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1)
