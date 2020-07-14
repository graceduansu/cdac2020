from deepscribe import DeepScribe
from disp_multiple_images import show_images

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def dict_to_np_arrays(img_data_save_path, label_data_save_path):
    """
    Process and save symbol dictionary generated from command line arguments as arrays in .npy files.

    Parameters:
        img_data_save_path (str): File path to save 3D numpy array of all image data to.
        label_data_save_path (str): File path to save 1D numpy array of all label data to.
    
    Returns:
        img_data (numpy.ndarray): 3D array of all image data from symbol dictionary.
        label_data (numpy.ndarray): 1D array of all label data corresponding to img_data.
    """
    
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)
    print("symbol_dict len (number of symbols):", len(symbol_dict))
    DeepScribe.transform_images(symbol_dict,
                                rescale_global_mean=True,
                                resize=100)

    img_data = []
    label_data = []

    for symb_name in symbol_dict:
        for symb_img in symbol_dict[symb_name]:
            img_data.append(symb_img.img)
            label_data.append(symb_name)

    label_encoder = LabelEncoder()
    print("label_data:", label_data)
    label_data = label_encoder.fit_transform(label_data)
    print("label_data encoded:", label_data)
    label_data = np.array(label_data, dtype='float32')
    print("label_data array shape:", label_data.shape)

    img_data = np.array(img_data, dtype='float32')
    print("img_data array shape:", img_data.shape)
    
    np.save(img_data_save_path, img_data)
    np.save(label_data_save_path, label_data)
    return img_data, label_data

def run_mnist(X, y, save_path):
    """
    Runs and evaluates model with MNIST architecture on OCHRE data.
 
    Parameters:
        X (numpy.ndarray): 3D array of all image data to be used for training and testing.
        y (numpy.ndarray): 1D array of all label data corresponding to img_data.
        save_path (str): File path to save Keras model to after training.
        
    Returns:
        history: Keras history object that holds records of metric values during training.
    """
    
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                              x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(232, activation=tf.nn.softmax)) #output_classes
    model.summary()

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    history = model.fit(x=x_train,y=y_train, epochs=100)

    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)

    model.save(save_path)

    print(history)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    
    #show_classification_report(X,y)
    return history

def show_classification_report(X, y, save_path):
    """
    RUN WITH -l max
    
    Displays classification report for a saved Keras model.
    
    Parameters:
        X (numpy.ndarray): 3D array of all image data used for training and testing.
        y (numpy.ndarray): 1D array of all label data corresponding to img_data.
        save_path (str): File path of Keras model to be evaluated.
    """
    
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    x_test = x_test.astype('float32')
    print('x_test shape:', x_test.shape)
    print("y_test:", y_test)
    
    """
    # Get symbol names
    symbol_names = list(DeepScribe.count_symbols())
    print("symbol names:", symbol_names)
    
    print("y_test:", y_test)

    y_test_list = []
    # Transform y values to corresponding label names
    for i in range(len(y_test)):
        s_idx = int(y_test[i])
        y_test_list.append(symbol_names[s_idx])
    
    print("y_test_list:", y_test_list)
    """
    
    try:
        model = keras.models.load_model(save_path) 
    except TypeError:
        print("Changing model binary file \"learning_rate\" to \"lr\"")
        f = h5py.File(save_path, "r+")
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

    """
    y_pred = model.predict_classes(x_test)
    print("y_pred:", y_pred)
    y_pred_list = []
    for i in range(len(y_pred)):
        s_idx = int(y_pred[i])
        y_pred_list.append(symbol_names[s_idx])
    
    print("y_pred_list:", y_pred_list)
    
    print(classification_report(y_test_list, y_pred_list))
    
    # Display 100 images
    # TODO: this does not run
    images = []
    titles = []
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

    i = 0
    while i in range(100):
        for layer in x_test:
            images.append(layer)
            title = "P: " + y_pred_list[i] + ", T: " + y_test_list[i]
            titles.append(title)
            i += 1

    show_images(images)
    """

def get_training_curve(output_file):
    """
    Display training loss and accuracy curves based on model training output text file.

    Parameters:
        output_file (str): File path of model training output text file.
    """

    f = open(output_file, "r")
    loss = []
    accuracy = []

    for line in f:
        l = line.find("loss: ")
        a = line.find("accuracy: ")
        if l != -1:
            loss.append(float(line[l+6:a].strip(' -')))
        if a != -1:
            accuracy.append(float(line[a+10:].strip()))
    
    print(loss)
    print(accuracy)
    x = list(range(1, 101))

    plt.plot(x, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST Model Training Loss on OCHRE")
    plt.show()

    plt.plot(x, accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Model Training Accuracy on OCHRE")
    plt.show()
    