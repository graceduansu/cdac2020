import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


def show_images(images, titles=None, color=False):  
    """
    Shows all images and titles in one matplotlib plot.
    
    Parameters:
        images (list): List of CV2 image arrays to be displayed.
        titles (list): List of titles corresponding to the images list.
        color (boolean): Displays images in color if True, otherwise displays in grayscale.
    """
    
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
        plt.axis("off")
        if color:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image)
        a.set_title(title, size=5)
    
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
