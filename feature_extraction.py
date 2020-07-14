import tensorflow as tf
from tensorflow import keras
import numpy as np

from deepscribe import DeepScribe

def extract_features(save_path):
    """
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)
    print("symbol_dict len (number of symbols):", len(symbol_dict))
    DeepScribe.transform_images(symbol_dict, gray=False, resize=100)
    img_data = []

    for symb_name in symbol_dict:
        for symb_img in symbol_dict[symb_name]:
            img_data.append(symb_img.img)

    img_data = np.array(img_data, dtype='float32')
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
    img_data = np.reshape(img_data, [img_data.shape[0], img_data.shape[1] * img_data.shape[2] * img_data.shape[3]])
    np.save("output/img_data_color.npy", img_data)
    """
    img_data = np.load("output/img_data_color1.npy")
    print(img_data.shape)

    x = keras.applications.vgg16.preprocess_input(img_data)
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    print("begin prediction:")
    features = model.predict(x, verbose=1)
    print(features)
    
    np.save(save_path, features)
