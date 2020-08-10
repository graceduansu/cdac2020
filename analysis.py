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
def show_reports(npz_file, encoding_file, model_path):
    """
    RUN WITH -l max
    
    Generates top 5 accuracy, classification report, confusion matrix, and 16 random incorrectly classified images for a saved Keras model.
    
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
    
    # Get symbol names
    symbol_names = np.load(encoding_file)
    print("symbol names:", symbol_names)
    
    y_test_list = []
    # Transform y values to corresponding label names
    for i in range(len(y_test)):
        s_idx = int(y_test[i])
        y_test_list.append(symbol_names[s_idx])
        
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
    pred_logits = model.predict(x_test)
    pred_labels = np.argmax(pred_logits, axis=1)

    successes = 0
    total = 0

    for i in range(len(y_test)):
        top_n_predictions = np.argpartition(pred_logits[i], -n)[-n:]

        if y_test[i] in top_n_predictions:
            successes += 1
        total +=1
    print(successes/total)

    y_pred_list = []
    # TODO: turn this into a function ?
    for i in range(len(pred_labels)):
        s_idx = int(pred_labels[i])
        y_pred_list.append(symbol_names[s_idx])

    print(classification_report(y_test_list, y_pred_list))

    ####### plot_confusion_matrix #################################
    confusion = confusion_matrix(y_test_list, y_pred_list)
    
    fig = plt.figure(figsize=(34, 34))
    ax = fig.add_subplot(111)
    plt.title("Confusion matrix")
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    ax.set_xticks(range(len(symbol_names)))
    ax.set_yticks(range(len(symbol_names)))
    # appending extra tick label to make sure everything aligns properly
    ax.set_xticklabels(symbol_names, rotation=45)
    ax.set_yticklabels(symbol_names)
    plt.tight_layout()
    filename = "records/confusion_matrix_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)
    
    ####### plot_incorrect_imgs #################################
    
    (incorrect_prediction_idx,) = np.not_equal(
        y_test, pred_labels
    ).nonzero()

    incorrect = np.array(incorrect_prediction_idx)

    _, axarr = plt.subplots(5, 5, figsize=(10, 10))

    for i, (ix, iy) in enumerate(np.ndindex(axarr.shape)):

        indx = incorrect[i]
        img = np.squeeze(x_test[indx, :, :])
        ground_truth = symbol_names[int(y_test[indx])]
        pred_label = symbol_names[int(pred_labels[indx])]
        h = entropy(pred_logits[indx, :])

        normalized = h / np.log(len(symbol_names))

        ax = axarr[ix, iy]
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(f"P:{pred_label},T:{ground_truth},H:{round(normalized, 2)}")
        ax.imshow(img, cmap="gray")

    filename = "records/incorrect_imgs_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)

def get_training_curves(output_file, epochs, n):
    """
    Generates training loss and accuracy curves based on model training output text file.

    Parameters:
        output_file (str): File path of model training output text file.
        epochs (int): Number of epochs to plot.
        n (int): Read first n lines of file only - up to the last line with training loss and accuracy data.
    """

    loss = []
    val_loss = []
    accuracy = []
    val_acc = []

    with open(output_file, "r", encoding='utf-8') as f:
        for i in range(n):
            line = next(f).strip()

            l = line.find("loss: ")
            vl = line.find("val_loss: ")
            a = line.find("accuracy: ")
            va = line.find("val_accuracy: ")
            if l != -1:
                loss.append(float(line[l+6:a].strip(' -')))
                accuracy.append(float(line[a+10:vl].strip(' -')))
                val_loss.append(float(line[vl+10:va].strip(' -')))
                val_acc.append(float(line[va+14:va+20]))
    
    print(loss)
    print(accuracy)
    x = list(range(1, epochs+1))

    plt.plot(x, loss, label="Training loss")
    plt.plot(x, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Training Loss")
    plt.legend(loc="upper right")
    filename = "records/training_loss_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)
    plt.close()

    plt.plot(x, accuracy, label="Training accuracy")
    plt.plot(x, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Training Accuracy")
    plt.legend(loc="upper left")
    filename = "records/training_accuracy_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)
