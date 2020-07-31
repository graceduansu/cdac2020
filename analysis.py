from deepscribe import DeepScribe

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import date
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy

# TODO: save model history and classification report as text files
def show_classification_report(npz_file, model_path):
    """
    RUN WITH -l max
    
    Displays top 5 accuracy and classification report for a saved Keras model.
    
    Parameters:
        npz_file (str): npz archive of train, valid, test data.
        model_path (str): File path of Keras model to be evaluated.
    """
    
    with np.load(npz_file) as data:
        x_test = data['x_test']
        y_test = data['y_test']

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    x_test = x_test.astype('float32')
    print('x_test shape:', x_test.shape)
    print("y_test:", y_test)
    
    # Get symbol names
    symbol_names = list(DeepScribe.count_symbols())
    print("symbol names:", symbol_names)
    
    y_test_list = []
    # Transform y values to corresponding label names
    for i in range(len(y_test)):
        s_idx = int(y_test[i])
        y_test_list.append(symbol_names[s_idx])
    
    print("y_test_list:", y_test_list)
    
    try:
        model = keras.models.load_model(model_path) 
    except TypeError:
        print("Changing model binary file \"learning_rate\" to \"lr\"")
        f = h5py.File(model_path, "r+")
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate","lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        print("File modification finished.")       

    # Print top 5 accuracy
    print("Top 5 accuracy:")
    n = 5
    probas = model.predict(x_test)
    print("probas shape:", probas.shape)

    successes = 0
    total = 0

    for i in range(len(y_test)):
        top_n_predictions = np.argpartition(probas[i], -n)[-n:]

        if y_test[i] in top_n_predictions:
            successes += 1
        total +=1
    print(successes/total)

    y_pred = model.predict_classes(x_test)
    print("y_pred:", y_pred)
    y_pred_list = []
    # TODO: turn this into a function ?
    for i in range(len(y_pred)):
        s_idx = int(y_pred[i])
        y_pred_list.append(symbol_names[s_idx])
    
    print("y_pred_list:", y_pred_list)
    
    print(classification_report(y_test_list, y_pred_list))
    
def plot_confusion_matrix(npz_file, model_path):
    """
    RUN WITH -l max
    
    Generates and saves confusion matrix for a saved Keras model.
    
    Parameters:
        npz_file (str): npz archive of train, valid, test data.
        model_path (str): File path of Keras model to be evaluated.
    """

    with np.load(npz_file) as data:
        x_test = data['x_test']
        y_test = data['y_test']

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    x_test = x_test.astype('float32')
    print('x_test shape:', x_test.shape)
    print("y_test:", y_test)
    
    # Get symbol names
    symbol_names = list(DeepScribe.count_symbols())
    
    y_test_list = []
    # Transform y values to corresponding label names
    for i in range(len(y_test)):
        s_idx = int(y_test[i])
        y_test_list.append(symbol_names[s_idx])
    
    print("y_test_list:", y_test_list)

    try:
        model = keras.models.load_model(model_path) 
    except TypeError:
        print("Changing model binary file \"learning_rate\" to \"lr\"")
        f = h5py.File(model_path, "r+")
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate","lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        print("File modification finished.")

    y_pred = model.predict_classes(x_test)
    print("y_pred:", y_pred)
    y_pred_list = []
    for i in range(len(y_pred)):
        s_idx = int(y_pred[i])
        y_pred_list.append(symbol_names[s_idx])
    
    print("y_pred_list:", y_pred_list)

    confusion = confusion_matrix(x_test, y_pred_list)
    
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    plt.title("Confusion matrix")
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    ax.set_xticks(range(len(symbol_names)))
    ax.set_yticks(range(len(symbol_names)))
    # appending extra tick label to make sure everything aligns properly
    ax.set_xticklabels(symbol_names)
    ax.set_yticklabels(symbol_names)
    filename = "records/confusion_matrix_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)

def plot_incorrect_imgs(npz_file, model_path):
    """
    Loads model, runs on test (?) data, picks 16 random incorrectly classified images.

    Parameters:
        npz_file (str): npz archive of train, valid, test data.
        model_path (str): File path of Keras model to be evaluated.
    """
    with np.load(npz_file) as data:
        x_test = data['x_test']
        y_test = data['y_test']

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    x_test = x_test.astype('float32')

    # Get symbol names
    symbol_names = list(DeepScribe.count_symbols())

    try:
        model = keras.models.load_model(model_path) 
    except TypeError:
        print("Changing model binary file \"learning_rate\" to \"lr\"")
        f = h5py.File(model_path, "r+")
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate","lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        print("File modification finished.")

    pred_logits = model.predict(x_test)
    pred_labels = np.argmax(pred_logits, axis=1)

    (incorrect_prediction_idx,) = np.not_equal(
        y_test, pred_labels
    ).nonzero()

    incorrect = np.array(incorrect_prediction_idx)

    _, axarr = plt.subplots(5, 5, figsize=(10, 10))

    for i, (ix, iy) in enumerate(np.ndindex(axarr.shape)):

        indx = incorrect[i]
        img = np.squeeze(x_test[indx, :, :])
        ground_truth = symbol_names[y_test[indx]]
        pred_label = symbol_names[pred_labels[indx]]
        h = entropy(pred_logits[indx, :])

        normalized = h / np.log(len(symbol_names))

        ax = axarr[ix, iy]
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(f"P:{pred_label},T:{ground_truth},H:{round(normalized, 2)}")
        ax.imshow(img, cmap="gray")

    filename = "records/incorrect_imgs_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)