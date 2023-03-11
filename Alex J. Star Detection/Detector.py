import os
import random, pandas as pd, cv2, numpy as np, csv
from PIL import Image

PATH = 'C:\\Users\\tomto\\Desktop\\FINAL\\'
DATA_PATH = PATH + 'data\\star\\'
DATA = pd.read_csv(PATH + '\\data.csv', sep=';')

IMG_HEIGHT = 32
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3

LABELS = [6, 8]
DATA_FILENAME = []
DATA_LABEL = []

for id, label in zip(DATA['id'], DATA['rays']):
    DATA_FILENAME.append(id)
    DATA_LABEL.append(label)

TEST_SIZE = 0.5

def balance_data(use_original_only:bool):
    """Clones transformed data, there is an even number of samples with different labels."""
    counter = DATA['rays'].value_counts()
    label_counter = {}

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

        if relevant:
            kernel.append(fname)

    # For each image, randomly rotate and move around, then save it with the same csv information.
    for k in range(label_counter[most_common] - label_counter[least_common]):
        rnd_id = str(random.sample(kernel, 1)[0])
        image = cv2.imread(DATA_PATH + rnd_id + ".jpg")

        # dividing height and width by 2 to get the center of the image
        height, width = image.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width / 2, height / 2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix
        theta = random.randint(0, 360)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle= theta, scale=1)

        # Randomly move image left to right
        w = int(width/8)
        h = int(height/8)
        a = random.randint(- w, w)
        b = random.randint(- h, h)
        trans_matrix = np.float32([[1, 0, a], [0 ,1, b]])

        # rotate the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        trans_image = cv2.warpAffine(src=rotated_image, M=trans_matrix, dsize=(width, height))

        # Get available name
        start = 1
        available = False
        fnames = os.listdir(DATA_PATH)
        while not available:

            ## Iterate over [aug1, aug2, ...] and get the lateast augx to save it
            curr_name = 'aug_6rays_' + str(start)
            available = curr_name + '.jpg' not in fnames
            start += 1

        # Read previous attributes (rays, daidem)
        with open(PATH+'data.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                id, rays, circle = row[0].split(';')
                if rnd_id == id:
                    break

        # Write new image id with same attributes
        with open(PATH+'data.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([curr_name, rays, circle])

        # Save image
        os.chdir(DATA_PATH)
        cv2.imwrite(curr_name + '.jpg', trans_image)

def get_data():
    """Returns a tuple of data values (tensor) and labels"""
    pass

def get_pre_trained_model():
    pass

def get_CNN_model():
    pass

def train_model(mdl):
    pass

if __name__ == '__main__':
    balance_data(use_original_only = True)
    exit(0)
    data_val, data_lab = get_data()
    X_train, X_val, Y_train, Y_val = train_test_split(data_val,
                                                      data_lab,
                                                      test_size = TEST_SIZE)
    model = get_CNN_model()
    train_model(mdl = model)