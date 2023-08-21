"""
Model Generation Library, usages:
* Training a model, with random hyper-parameters*, saving it according to the results.
* Confusion matrix - visualizing number of mistakes by label.
* History graph - plot loss and accuracy history.

** Hyper-parameters which are randomized are -
   - Number of convolutions      (Mainly a lower number, one to three layers)
   - Filter amount of each layer (4 - 128 by jumps of times 2)
   - Size of each filter (Mainly 3)
   - Activation function (Mainly RELU)
   - Opimizer            (Mainly RMSProp)

Randomness was reduced after analyzing results. Shallow models with a low amount of convolutions,
Low filter size and a known activation function (ReLU, as opposed to ELU or GELU) succeeded more.
"""

import os

import keras.activations
from Data import *
from Model import *
from Cons import *

DATA = pd.read_csv(os.getcwd() + '\\data.csv', sep=SEP)
for id, label in zip(DATA['id'], DATA['rays']):
    DATA_FILENAME.append(id)
    DATA_LABEL.append(label)

"""
    Class weights - each class penalizes for a mistake. But a class with less samples (e.g the six-rays)
    will bear a heavier penalty for a mistake.
"""
weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced',
                                                          classes = np.unique(DATA_LABEL),
                                                          y = DATA_LABEL)
weights = {i : weights[i] for i in range(len(weights))} # Convert to a dictionary to be compatible with keras.


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


def confusion_matrix_clf(model, test = False, show = True, log = False):
    """
    
    :param model: Model to analyze
    :param test: TRUE - will take samples from test directory, FALSE - will take from training
    :param show: Plot the confusion matrix
    :param log: Output to console the results with probabiltiies from the softmax activation
    :return: 
    """TEST = []
    x, y = get_data(test, TEST)
    src = TEST if test else DATA_FILENAME

    if len(x) != len(y) or len(x) == 0: return

    predictions_prob = model.predict(tf.convert_to_tensor(x), verbose=0)

    predictions = [LABELS[i] for i in predictions_prob.argmax(axis=-1)]
    real = [LABELS[i.index(True)] for i in y]

    if log:
        print(f'{decor} Predictions strength {decor} \n{predictions_prob}')
        print(f'{decor} Predictions label    {decor} \n{predictions}')
        print(f'{decor} Actual label         {decor} \n{real}')

    wrong_fnames = []
    for i in range(len(predictions)):
        if predictions[i] != real[i]:
            wrong_fnames.append(src[i])


    if show:
        cm = confusion_matrix(real, predictions)
        unique = set(real + predictions)
        unique = sorted(unique)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique)
        disp.plot()
        plt.show()

    # if len(wrong_fnames) == 0:

    print(f'Mistakes - {len(wrong_fnames)} / {len(predictions)}')
    # Return a list of wrong gusses
    return wrong_fnames


def show_images(img_list, test = False):
    """
    :param img_list: List of images to be shown one after the other
    :param test: Take from test directory or training directory
    :return:
    """
    path = TEST_PATH if test else DATA_PATH
    for img in img_list:
        curr_dir = path + str(get_label(img)) + "\\"
        plt.imshow(cv2.imread(curr_dir + img + ".jpg"))
        plt.title("File " + img + " , Label:" + str(get_label(img)))
        plt.show()


def model_generator():
    """
    From i = 0 to n does:
        Generates a model with random parameters.
        Runs the model for a defined amount of epochs.
        Tests the model on untrained images.
        Outputs information to screen.
        Saves the model if it predicted all the test images correctly.
    :return: Amount of saved models overall, iterations ran
    """
    conv_nums = [1, 2, 3] #, 4, 5, 7, 8]
    fil_nums = [2, 4, 8, 16, 32, 64, 128]
    fil_sizes = [3, 5, 7, 9]
    opts = ['rmsprop', 'ftrl', 'adam', 'sgd']
    activs = ['relu', keras.activations.leaky_relu, keras.activations.relu6, 'swish']


    runs = 10000
    counter = len(os.listdir(MODELS_PATH)) ## Count how many models predicted all test photos successfully.

    print(f'Starting search for 0-test-mistakes models.')
    for i in range(runs):
        ## Move random files from train to test

        move_train_to_dir(1, [0, 6, 8])

        conv_num = random.sample(conv_nums, 1)[0]
        fil_num = random.sample(fil_nums, 1)[0]
        fil_size = random.sample(fil_sizes, 1)[0]
        opt = random.sample(opts, 1)[0]
        activ = random.sample(activs, 1)[0]

        print(f'conv_num = {conv_num}, fil_num = {fil_num},'
              f' fil_size = {fil_size}, opt = {opt}, activ = {activ}')

        model = get_CNN_model(conv_num = conv_num, fil_num = fil_num,
                              fil_size = fil_size, opt = opt, activ = activ)

        history = train_model(model, w = weights)
        failures = confusion_matrix_clf(model, test = True, show = False, log = True)

        if len(failures) == 0:
            model.save(MODELS_PATH + str(counter) + ".h5")
            print(f'Model number {counter} saved.')
            counter += 1

        ## Move all test samples back, to shuffle and transfer other samples to the test directory.
        move_all_back([0, 6, 8])

    return counter, runs


if __name__ == '__main__':
    move_all_back(LABELS) ## Run again - changed: batch size 64, weights,
    counter, runs = model_generator()
    print(f'{counter} models saved in {runs} runs.')

