import tensorflow.python.framework.errors_impl
from ModelGen import *



def predict_label(model, filename):

    img = tf.io.read_file(filename)
    img = decode_img(img)
    img = tf.convert_to_tensor(img)
    img = np.expand_dims(img, axis = 0) ## (H, W, C) -> (None, H, W, C)

    res = model.predict(img, verbose=0)
    ans = res.argmax(axis=-1)[0]

    return LABELS[ans]


def load_models(md_list):
    ans = []
    for md in md_list:
        if 'ignore' in md: continue
        ans.append(keras.models.load_model(MODELS_PATH + str(md)))
    return ans



def get_all_predictions(loaded_models, filename):
    """Input - list of models, filename
       Returns - Dictionary, key - label, value - how many models predicted that key"""
    predictions = { label : 0 for label in LABELS }     ## Count occurence of each prediction
    for i, md in enumerate(loaded_models):

        guess = predict_label(md, filename)
        predictions[guess] += 1
        print(f'.', end="")                             ## Show progress, can be commented out

    print("")
    return predictions


def pred_conf(filepath : str, loaded_models : list):
    try:
        print(f'Predicting', end="")  ## end="" dosen't add a new line
        preds = get_all_predictions(loaded_models, filepath)

        max_label = max(preds, key=preds.get)  ## Get key K with most predictions
        max_val = preds[max_label]  ## with K get value to calculate confidence
        conf = max_val / len(loaded_models)

        return max_label, conf

    except tensorflow.errors.NotFoundError:
        print(f'\nFile not found.')
        return -1, -1



if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')                   ## Disable warning of using predict in for loop

    print("Loading all models.")
    models = os.listdir(MODELS_PATH)
    loaded_models = load_models(models)

    while True:

        print(f'Enter a file name if it is in the same directory, or the path.')
        filename = input()
        filename = filename if '.' in filename else filename + '.jpg'

        guess, conf = pred_conf(filename, loaded_models)
        print(f'Label - {guess}, Confidence = {conf * 100:.2f}%')

