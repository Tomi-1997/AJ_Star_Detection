PATH_TOM = 'C:\\Users\\tomto\\Desktop\\FINAL\\'
PATH_SON = 'D:\\UNI\\FinalProject\\'
PATH = PATH_SON
# CSV_PATH =
DATA_PATH = PATH + 'data\\star_side\\'
TEST_PATH = PATH + 'data\\test\\'
MODELS_PATH = PATH + 'models\\'

SEP = ';'

IMG_H = 64
IMG_W = IMG_H
CHANNELS = 3

LABELS = [0, 6, 8]
DATA_FILENAME = []
DATA_LABEL = []
TEST = []

TRAIN_SIZE = 0.7
BATCH_SIZE = 32
EPOCHS = 300

decor = "♦♦♦"

import keras
import sklearn
import keras.utils
from keras.utils import to_categorical
from keras import regularizers

from keras import layers, Model
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from keras.layers import Input, GlobalMaxPooling2D, Dropout, Dense, Flatten

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf

import os
import math
import csv
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
