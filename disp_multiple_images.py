import matplotlib.pyplot as plt
import math
import numpy as np

def show_images(images, titles=None):

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    cols = math.floor(math.sqrt(n_images))

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, math.ceil(n_images / float(cols)), n + 1)
        
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()