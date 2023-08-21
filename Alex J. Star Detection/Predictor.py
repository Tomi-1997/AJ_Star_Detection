"""
Prediction Library, usages:
* Get full prediction and confidence for an unknown image.

It is done by-
- Loading all models from the '/models/*.h5' directory.
- Count predictions from each model
- Return label with most predictions
- Calculate confidence by amount of models which predicted that label, divided by overall models.
(e.g. out of 30 models, 20 models predict the label is 8- the outcome is 8 with 66% confidence.)
"""

import time
import tensorflow.keras as keras
import tensorflow as tf
from Cons import LABELS, MODELS_PATH, IMG_W, IMG_H, CHANNELS
import os
import numpy as np


def predict_label(model, filename, is_path=True):
    """

    :param model: Model to predict label of image
    :param filename: Image object \ path, depends on the next variable
    :param is_path:
    :return: Prediction of a given image
    """
    img = None

    # Treat filename as a path
    if is_path:
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=CHANNELS)

    # Treat filename is an already open image if is_path is false
    else:
        img = filename

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[IMG_H, IMG_W])
    img = tf.convert_to_tensor(img)

    img = img[:, :, :3] # Slice 4th channel which some extension have

    img = np.expand_dims(img, axis=0)  ## (H, W, C) -> (None, H, W, C)

    res = model.predict(img, verbose=0)
    ans = res.argmax(axis=-1)[0]

    return LABELS[ans]


def load_models(md_list, path):
    """

    :param md_list: List of model strings
    :param path: Path of models to load from
    :return: a list of loaded models
    """
    ans = []
    for md in md_list:
        if not md[0].isdigit(): continue
        # ans.append(keras.models.load_model(MODELS_PATH + str(md)))
        # print(path + str(md))
        ans.append(keras.models.load_model(path + str(md)))
    return ans


def get_all_predictions(loaded_models, filename, is_path=True):
    """Input - list of models, filename
       Returns - Dictionary, key - label, value - how many models predicted that key"""
    predictions = {label: 0 for label in LABELS}  ## Count occurence of each prediction
    for i, md in enumerate(loaded_models):
        guess = predict_label(md, filename, is_path)
        predictions[guess] += 1
        # print(f'(md={i}, guess={guess})', end="") ## Show prediciton of each model, can be commented out
        print(f'.', end="")  ## Show progress, can be commented out

    print("")
    return predictions


def pred_conf(filepath, loaded_models: list, is_path=True):
    """
        Filepath - A file path from disk, or an image. If it is an image, make sure is_path is false
       Loaded_model - List of loaded h5 models
       is_path - True if the filepath is indeed a path to be opened, or false if an image (To save writing to disk) is given.
    """
    try:
        print(f'Predicting', end="")  ## end="" dosen't add a new line
        preds = get_all_predictions(loaded_models, filepath, is_path)

        max_label = max(preds, key=preds.get)  ## Get key K with most predictions
        max_val = preds[max_label]  ## with K get value to calculate confidence
        conf = max_val / len(loaded_models)

        return max_label, conf

    except tf.errors.NotFoundError:
        print(f'\nFile not found.')
        return -1, -1


def save_as_H5(md_list):
    """
        Incase of models saved not as H5 (More efficient format of models, faster to load and takes less space)
        Loads not H5 models and saves as H5.
    :param md_list:
    :return:
    """
    for md in md_list:
        if 'ignore' in md: continue
        model = keras.models.load_model(MODELS_PATH + str(md))
        model.save(MODELS_PATH + "\\H05\\" + str(md) + ".h5")


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')  ## Disable warning of using predict in for loop

    print("Loading all models.")
    models = os.listdir(MODELS_PATH)
    loaded_models = load_models(models, path=MODELS_PATH)

    while True:
        print(f'Enter a file name if it is in the same directory, or the path.')
        filename = input()
        filename = filename if '.' in filename else filename + '.jpg'

        guess, conf = pred_conf(filename, loaded_models)
        print(f'Label - {guess}, Confidence = {conf * 100:.2f}%')