import numpy as numpy


class Symbol_Image:
    def __init__(self, symb_name, uuid, img):
        self.symb_name = symb_name
        self.uuid = uuid
        self.img = img

    def __str__(self):
        #maybe print img size
        return "UUID: %s | symbol name: %s" % (self.uuid, self.symb_name)
