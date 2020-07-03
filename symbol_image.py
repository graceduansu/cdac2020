import numpy as numpy


class Symbol_Image:
    """
    A class representing a single image of a symbol.

    Attributes:
        symb_name (string): The name of the symbol shown inside the image, extracted from the image's filename.
        uuid (string): The universally unique ID of the item represented by the cuneiform sign in the OCHRE database.
        img (numpy array): The image data, stored in a numpy array.
    """
    def __init__(self, symb_name, uuid, img):
        self.symb_name = symb_name
        self.uuid = uuid
        self.img = img

    def __str__(self):
        # maybe print img size
        return "UUID: %s | symbol name: %s" % (self.uuid, self.symb_name)
