## Generates many models and saved the best.
import os

from Data import *
from Model import *
from Cons import *

DATA = pd.read_csv(os.getcwd() + '\\data.csv', sep=SEP)
for id, label in zip(DATA['id'], DATA['rays']):
    DATA_FILENAME.append(id)
    DATA_LABEL.append(label)

# Compute class weights and turn into a dict.
weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced',
                                                          classes = np.unique(DATA_LABEL),
                                                          y = DATA_LABEL)
weights = {i : weights[i] for i in range(len(weights))}


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


def predict(model, fname, star = True, test = False):
    test_img = [image_to_tensor(fname, test)]
    res = model.predict(tf.convert_to_tensor(test_img), verbose=0)
    ans = res.argmax(axis=-1)[0]
    print(res)
    print(f'Guess : {LABELS[ans]} rays.')


def confusion_matrix_clf(model, test = False, show = True, log = False):
    """Test - False
        Makes a prediction on the whole data set from 'data.csv' and plots a matrix depicting the model's choices
        Test - True
        Makes prediction on all images from test folders."""

    TEST = []
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
    path = TEST_PATH if test else DATA_PATH
    for img in img_list:
        curr_dir = path + str(get_label(img)) + "\\"
        plt.imshow(cv2.imread(curr_dir + img + ".jpg"))
        plt.title("File " + img + " , Label:" + str(get_label(img)))
        plt.show()


def model_generator():
    conv_nums = [1, 2] #, 3, 4, 5, 7, 8]
    fil_nums = [2, 4, 8, 16, 32, 64]
    fil_sizes = [3, 5]
    opts = ['rmsprop', 'ftrl', 'adam', 'sgd']
    activs = ['relu', 'elu', 'selu', 'gelu']


    runs = 10000
    counter = len(os.listdir(MODELS_PATH)) ## Count how many models predicted all test photos successfully.

    print(f'Starting search for 0-test-mistakes models.')
    for _ in range(runs):
        ## Move random files from train to test
        move_train_to_dir(2, [0, 6, 8])

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
            model.save(MODELS_PATH + str(counter))
            counter += 1

        del model
        ## Move them all back
        move_all_back([0, 6, 8])
        # show_images(failures, test = True)

    return counter, runs

# def get_model_summary(range):
#     for i in range(runs):
#         model = keras.models.load_model(MODELS_PATH + str(i))
#         model.summary1()

if __name__ == '__main__':
    move_all_back(LABELS)
    counter, runs = model_generator()
    print(f'{counter} models saved in {runs} runs.')

