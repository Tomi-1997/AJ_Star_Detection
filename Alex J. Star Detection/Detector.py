import pandas as pd
PATH = 'C:\\Users\\tomto\\Desktop\\FINAL\\'
DATA_PATH = PATH + 'data\\star\\'
DATA = pd.read_csv(PATH + '\\data.csv')

IMG_HEIGHT = 32
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3

LABELS = [6, 8]

def balance_data():
    """Clones data, while rotating it until there is an even number of samples with different labels."""
    counter = DATA['label'].value_counts()
    label_counter = {}
    for lbl in counter.keys():
        if lbl in LABELS:
            label_counter[lbl] = counter[lbl]

    most_common = max(label_counter)
    least_common = min(label_counter)
    while label_counter[most_common] > label_counter[least_common]:
        # Save rotated images
        break

if __name__ == '__main__':
    balance_data()