import numpy as np
from deepscribe import DeepScribe
import mnist


def run():
    # Code to test DeepScribe functions:
    # symbol_dict = {}
    # DeepScribe.load_images(symbol_dict)
    # for s in symbol_dict.values():
    #     for symb_img in s:
    #         print(symb_img)
    # DeepScribe.transform_images(symbol_dict)
    # DeepScribe.display_images(symbol_dict)
 
    # Code to build model:
    # X, y = mnist.dict_to_np_arrays() # run with -l max for all symbols

    X = np.load("output/img_data1.npy")
    y = np.load("output/label_data1.npy")
    mnist.run_mnist(X, y)
    # mnist.show_classification_report(X, y)


if __name__ == '__main__':
    run()