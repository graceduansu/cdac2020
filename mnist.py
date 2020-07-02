from deepscribe import DeepScribe

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def dict_to_np_arrays(symbol_dict):
    DeepScribe.transform_images(symbol_dict,
                                rescale_global_mean=True,
                                resize=100)

    img_data = []
    label_data = []
    #TODO: is list append or np append better?
    for symb_name in symbol_dict:
        for symb_img in symbol_dict[symb_name]:
            img_data.append(symb_img.img)
            label_data.append(symb_name)

    #TODO: does dtype of array matter? This is default float64
    img_data = np.array(img_data, dtype='float32')
    
    label_encoder = LabelEncoder()
    label_data = label_encoder.fit_transform(label_data)
    label_data = np.array(label_data, dtype='float32')
    print(img_data.shape)
    print(label_data.shape)
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
    #img_data = np.reshape(img_data,[img_data.shape[0],img_data.shape[1]*img_data.shape[2]*img_data.shape[3]])
    np.save('img_data', img_data)
    np.save('label_data', label_data)
    return img_data, label_data


def run_mnist(X, y):
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
    model.add(Dense(278, activation=tf.nn.softmax))
    model.summary()

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)

    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)

    model.save("MNIST_model_on_OCHRE")

    image_index = 10
    plt.imshow(x_test[image_index].reshape(100,100), cmap='Greys')
    pred = model.predict(x_test[image_index].reshape(1, 100, 100, 1))
    print('Label: ', y_test[image_index])
    print(pred)
    print(pred.argmax())
