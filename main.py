import numpy as np
from deepscribe import DeepScribe
import mnist
import feature_extraction
from matplotlib import pyplot


def run():
    DeepScribe.count_symbols(sort_by_freq=True)
    # Code to build model:
    # X, y = mnist.dict_to_np_arrays("output/img_data1.npy", "output/label_data1.npy") # run with -l max for all symbols

    # X = np.load("output/img_data_gray.npy")
    # y = np.load("output/label_data_gray.npy")
    # mnist.run_mnist(X, y, "output/MNIST_model_on_OCHRE_100_epochs")
    # mnist.show_classification_report(X, y, "output/MNIST_model_on_OCHRE_100_epochs")

    # mnist.get_training_curve("training_output_100_epochs.txt")

    # Code to test feature extraction:
    # feature_extraction.extract_features("output/vgg16_fts_color.npy")

    # Code to show feature maps - does not work
    # X = np.load("output/vgg16_fts_color.npy")
    # print(X.shape)
    # for fmap in X:
    #     ix = 1
    #     for i in range(8):
    #         for i in range(8):
    #             ax = pyplot.subplot(8,8,ix)
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             pyplot.imshow(fmap[0,:,:])
    #             ix += 1
    # pyplot.show()

    # y = np.load("output/label_data_gray.npy")
    # feature_extraction.random_forest(X, y)

    # feature_extraction.neural_net()


if __name__ == '__main__':
    run()
