from deepscribe import DeepScribe
from disp_multiple_images import show_images

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Importing the required Keras modules containing model and layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def dict_to_np_arrays():
    """
    Creates dictionary of symbols based on command line arguments, resizes 
    images, rescales image values, encodes labels, and saves image and 
    label data as numpy arrays in .np files.
    """
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)
    print("symbol_dict len (number of symbols):", len(symbol_dict))
    DeepScribe.transform_images(symbol_dict,
                                rescale_global_mean=True,
                                resize=100)

    img_data = []
    label_data = []
    # TODO: is list append or np append better?
    for symb_name in symbol_dict:
        for symb_img in symbol_dict[symb_name]:
            img_data.append(symb_img.img)
            label_data.append(symb_name)

    # TODO: does dtype of array matter? This is default float64
    label_encoder = LabelEncoder()
    print("label_data:", label_data)
    label_data = label_encoder.fit_transform(label_data)
    print("label_data encoded:", label_data)
    label_data = np.array(label_data, dtype='float32')
    print("label_data array shape:", label_data.shape)

    img_data = np.array(img_data, dtype='float32')
    print("img_data array shape:", img_data.shape)
    
    # For color images: 
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
    # img_data = np.reshape(img_data,[img_data.shape[0],img_data.shape[1]*img_data.shape[2]*img_data.shape[3]])
    
    np.save('output/img_data1', img_data)
    np.save('output/label_data1', label_data)
    return img_data, label_data

# TODO: add training curve?
def run_mnist(X, y):
    """
    Runs and evaluates model with MNIST architecture on OCHRE data.

    Arguments:
    X (numpy array): All image data to be used for training and testing.
    y (numpy array): All label data for each image.
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

    # output_classes = DeepScribe.count_symbols()

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
    model.fit(x=x_train,y=y_train, epochs=100)

    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)

    model.save("output/MNIST_model_on_OCHRE_100_epochs")

    # show_classification_report(X,y)

# TODO: Add filename params
# TODO: Add confusion matrix
def show_classification_report(X,y):
    """
    MUST RUN WITH -l max
    """
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    x_test = x_test.astype('float32')
    print('x_test shape:', x_test.shape)
    # print("indices:", indices)
    print("y_test:", y_test)
    
    # Get symbol names
    symbol_names = list(DeepScribe.count_symbols())
    print("symbol names:", symbol_names)

    model = keras.models.load_model("output/MNIST_model_on_OCHRE1")
    y_pred = model.predict_classes(x_test)
    
    # Get LabelEncoder, then inverse transform y values to corresponding label names
    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_names)
    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred)
    print("y_test:", y_test)
    print("y_pred:", y_pred)

    # Call sklearn classification function
    print("y_test dtype:", y_test.dtype)
    print("y_pred dtype:", y_pred.dtype)
    print(classification_report(y_test, y_pred))
    
    # Display 100 images
    images = []
    titles = []
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
    """
    while i in range(100):
        for layer in x_test:
            images.append(layer)
            title = "Pred: " + 
            i += 1
    """
    show_images(images)