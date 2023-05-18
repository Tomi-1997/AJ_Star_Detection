from ModelGen import *

def predict_label(model, filename):

    img = tf.io.read_file(filename)
    img = decode_img(img)
    img = tf.convert_to_tensor(img)
    img = np.expand_dims(img, axis = 0) ## (H, W, C) -> (None, H, W, C)

    res = model.predict(img, verbose=0)
    ans = res.argmax(axis=-1)[0]

    return LABELS[ans]

if __name__ == '__main__':



    while True:

        print(f'Hi, please enter a file name if it is in the same directory, or the path.')
        filename = input()
        print(f'Predicting', end = "")                      ## end="" dosen't add a new line
        models = os.listdir(MODELS_PATH)

        tf.get_logger().setLevel('ERROR')                   ## Disable warning of using predict in for loop
        predictions = { label : 0 for label in LABELS }     ## Count occurence of each prediction

        for i in models:
            model = keras.models.load_model(MODELS_PATH + str(i))
            guess = predict_label(model, filename)

            predictions[guess] += 1
            print(f'.', end="")

        print(predictions)
