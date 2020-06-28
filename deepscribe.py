import cv2
import numpy as np
from matplotlib import pyplot as plt

from symbol_image import Symbol_Image
import argparse
from glob import glob, iglob
from disp_multiple_images import show_images


class DeepScribe:
    @staticmethod
    def get_command_line_args():
        #TODO: Why is there no "Error: Too few arguments"?
        #TODO: create default for symbol
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--symbol', help='symbol query')

        #no limit indicated as 'max'
        parser.add_argument('-l', '--limit', help='limit on number of image results', default=100)
        args = parser.parse_args()
        return args

    @staticmethod
    def load_images(symbol_dict):
        args = DeepScribe.get_command_line_args()
        query = "a_pfa/" + args.symbol + "_*.jpg"
        count = 0
        
        for fn in iglob(query):
            separator_idx = len("a_pfa/" + args.symbol + "_")
            extension_idx = fn.rfind(".jpg")
            name = args.symbol
            uuid = fn[separator_idx:extension_idx]
            symb_img = Symbol_Image(name, uuid, cv2.imread(fn))
            #print(symb_img)
            
            if args.symbol in symbol_dict:
                symbol_dict[args.symbol].append(symb_img)
            else:
                symbol_dict[args.symbol] = [symb_img]
            count += 1

            if args.limit != 'max':
                if count >= args.limit:
                    break

    @staticmethod
    def display_images(symbol_dict):
        # show_images()
        return

    @staticmethod
    def transform_images():
        return
