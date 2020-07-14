import numpy as np
from deepscribe import DeepScribe
import mnist
import feature_extraction


def run():
    # Code to build model:
    # X, y = mnist.dict_to_np_arrays("output/img_data1.npy", "output/label_data1.npy") # run with -l max for all symbols

    # X = np.load("output/img_data1.npy")
    # y = np.load("output/label_data1.npy")
    # mnist.run_mnist(X, y, "output/MNIST_model_on_OCHRE_100_epochs")
    # mnist.show_classification_report(X, y, "output/MNIST_model_on_OCHRE_100_epochs")

    # mnist.get_training_curve("training_output_100_epochs.txt")

    # Code to test feature extraction:
    feature_extraction.extract_features("output/vgg16_fts_color.npy")

if __name__ == '__main__':
    run()
