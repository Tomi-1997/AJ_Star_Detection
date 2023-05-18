from Detector import *

if __name__ == '__main__':

    ## Test all saved models on all of the data.
    ## (Models were saved if they scored no errors or test label 6 or 8)
    models = os.listdir(MODELS_PATH)
    for i in models:
        model = keras.models.load_model(MODELS_PATH + str(i))
        failures = confusion_matrix_clf(model, test = False, show = True, log = False)

        if len(failures) < 20:
            show_images(failures, test = False)
