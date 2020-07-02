import numpy as np
from deepscribe import DeepScribe
import mnist


def run():
    # symbol_dict = {}
    # DeepScribe.count_symbols() #result: 278 symbols, 110230 images
    
    # DeepScribe.load_images(symbol_dict)

    # for s in symbol_dict.values():
    #     for symb_img in s:
    #         print(symb_img)

    # DeepScribe.transform_images(symbol_dict, resize=100)
    # DeepScribe.display_images(symbol_dict)
    
    # X, y = mnist.dict_to_np_arrays(symbol_dict)
    
    X = np.load("output/img_data.npy")
    y = np.load("output/label_data.npy")
    mnist.run_mnist(X, y)


if __name__ == '__main__':
    run()