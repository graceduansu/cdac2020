from deepscribe import DeepScribe

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import date
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from blocks import conv_block, identity_block

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_x_y_arrays(img_data_save_path, label_data_save_path, n=50):
    """
    Process and save symbol dictionary generated from command line arguments as arrays in .npy files.

    Parameters:
        img_data_save_path (str): File path to save 3D numpy array of all image data to.
        label_data_save_path (str): File path to save 1D numpy array of all label data to.
        n (int): Get top n most frequently occurring signs in dataset. Set to "all" to get all signs.
    
    Returns:
        img_data (numpy.ndarray): 3D array of all image data from symbol dictionary.
        label_data (numpy.ndarray): 1D array of all label data corresponding to img_data.
    """
    
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)
    print("symbol_dict len (number of symbols):", len(symbol_dict))

    # Taking only top n signs
    if n != "all":
        top_n_sign_list = DeepScribe.count_symbols(sort_by_freq=True)
        for s in symbol_dict:
            if s not in top_n_sign_list:
                symbol_dict.pop(s) 
        print("symbol_dict len after selecting top n:", len(symbol_dict))

    DeepScribe.transform_images(symbol_dict,
                                gray=True,
                                resize=50)

    img_data = []
    label_data = []

    for symb_name in tqdm(symbol_dict, desc='symbol names'):
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
    
    filename = "output/label_encoding_" + date.today().strftime("%m%d%y")
    classes = label_encoder.classes_
    print("classes:", classes)
    np.save(filename, classes)
    return img_data, label_data, classes

def train_test_val_split(X, y, fracs=(0.7, 0.1, 0.2)):
    # TODO: add stratification param?
    """
    Splits data into training, testing, and validation sets, WITH STRATIFICATION and saves them into a .npz archive.
 
    Parameters:
        X (numpy.ndarray): 3D array of all image data to be used for training and testing.
        y (numpy.ndarray): 1D array of all label data corresponding to img_data.
        fracs (list): list of decimals indicating train, validation, and test proportions, in that order.
        
    Returns:
        x_train, x_valid, x_test, y_train, y_valid, y_test
    """

    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=fracs[2],
                                                        stratify=y)
    
    valid_frac = fracs[1] / (fracs[0] + fracs[1])                                                                                                        
    x_train, x_valid, y_train, y_valid = train_test_split(
                x_train, 
                y_train, 
                test_size=valid_frac,
                stratify=y_train
            )
    
    filename = "output/split_data" + date.today().strftime("%m%d%y")
    np.savez_compressed(
            filename,
            x_train=x_train,
            x_test=x_test,
            x_valid=x_valid,
            y_train=y_train,
            y_test=y_test,
            y_valid=y_valid,
        )

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def build_resnet18(input_shape, num_classes):
    """
    Builds and returns ResNet18 model with Keras.
 
    Parameters:
        input_shape (tuple): Dimensions of model input.
        num_classes (int): Number of prediction classes the model should output.
        
    Returns:
        model (keras.model)
    """

    # set up regularizer
    reg_l1 = 0.0  # default param - no penalty
    reg_l2 = 0.0  # default param - no penalty
    regularizer = keras.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)

    img_input = keras.layers.Input(shape=input_shape)
    input_dropout = keras.layers.Dropout(0.0)(img_input)
    x = keras.layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(input_dropout)
    x = keras.layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding="valid",
        kernel_initializer="he_normal",
        name="conv1",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(input_dropout)
    x = keras.layers.BatchNormalization(name="bn_conv1")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # default values from the original ResNet50 implementation.
    x = conv_block(
        x,
        3,
        [64, 64, 256],
        stage=2,
        block="a",
        strides=(1, 1),
        regularizer=regularizer,
    )
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", regularizer=regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", regularizer=regularizer)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", regularizer=regularizer)
    x = identity_block(
        x, 3, [128, 128, 512], stage=3, block="b", regularizer=regularizer
    )
    x = identity_block(
        x, 3, [128, 128, 512], stage=3, block="c", regularizer=regularizer
    )

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)

    for _ in range(3): #param: n_dense layers
        x = keras.layers.Dense(512, activation="relu")(x)

    predictions = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=img_input, outputs=predictions)
    return model

def train(X, y, model_path):
    """
    Runs and evaluates model with ? architecture on OCHRE data.
 
    Parameters:
        X (numpy.ndarray): 3D array of all image data to be used for training and testing.
        y (numpy.ndarray): 1D array of all label data corresponding to img_data.
        model_path (str): File path to save Keras model to after training.
        
    Returns:
        history: Keras history object that holds records of metric values during training.
    """
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = train_test_val_split(X, y)

    # GRAY: Copy image values into 3 channels so that it can work with the Keras API
    # x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
    
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                              x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2],
                            1)    
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                            1)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_valid', x_valid.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    num_classes = len(DeepScribe.count_symbols())
    # TODO: add "architecture" parameter ?
    model = build_resnet18(input_shape, num_classes)
    model.summary()

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc")])

    callbacks = [keras.callbacks.TerminateOnNaN()]
    callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        )
    callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", min_delta=1e-4, patience=5
            )
        )

    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=128,
                        steps_per_epoch=x_train.shape[0] / 32,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks)

    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)

    model.save(model_path)

    print(history)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = "records/model_accuracy_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    filename = "records/model_loss_" + date.today().strftime("%m%d%y") + ".png"
    plt.savefig(filename)
    
    return history
