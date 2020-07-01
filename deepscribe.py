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
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--symbol', help='symbol query')

        #no limit should be input as 'max'
        parser.add_argument('-l',
                            '--limit',
                            help='limit on number of image results',
                            default=100)
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

            #not using cv2.imread() in order to read unicode filenames
            img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8),
                               cv2.IMREAD_UNCHANGED)
            symb_img = Symbol_Image(name, uuid, img)

            if args.symbol in symbol_dict:
                symbol_dict[args.symbol].append(symb_img)
            else:
                symbol_dict[args.symbol] = [symb_img]
            count += 1

            if args.limit != 'max':
                if count >= args.limit:
                    break

    @staticmethod
    def display_images(symbol_dict, color=False):
        images = []

        for s in symbol_dict.values():
            for symb_img in s:
                images.append(symb_img.img)

        show_images(images, color=color)

    #global_thresh can be a threshold value or -1 for none
    @staticmethod
    def transform_images(symbol_dict,
                         gray=True,
                         gauss_filter=-1,
                         bilat_filter=-1,
                         global_thresh=-1,
                         adapt_thresh_mean=-1,
                         adapt_thresh_gauss=-1,
                         otsus=-1,
                         laplacian=False,
                         canny=-1,
                         rescale_global_mean=False,
                         resize=-1):
        for s in symbol_dict.values():
            for symb_img in s:
                if gray:
                    gray_img = cv2.cvtColor(symb_img.img, cv2.COLOR_BGR2GRAY)
                    symb_img.img = gray_img
                if gauss_filter != -1:
                    blur_img = cv2.GaussianBlur(symb_img.img,
                                                (gauss_filter, gauss_filter),
                                                0)
                    symb_img.img = blur_img
                if bilat_filter != -1:
                    bilat_img = cv2.bilateralFilter(symb_img.img,
                                                    bilat_filter[0],
                                                    bilat_filter[1],
                                                    bilat_filter[2])
                    symb_img.img - bilat_img
                if global_thresh != -1:
                    ret, thresh_img = cv2.threshold(symb_img.img,
                                                    global_thresh, 255,
                                                    cv2.THRESH_BINARY)
                    symb_img.img = thresh_img
                if adapt_thresh_mean != -1:
                    thresh_img = cv2.adaptiveThreshold(symb_img.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY, adapt_thresh_mean, 2)
                    symb_img.img = thresh_img
                if adapt_thresh_gauss != -1:
                    thresh_img = cv2.adaptiveThreshold(symb_img.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY, adapt_thresh_gauss, 2)
                    symb_img.img = thresh_img
                if otsus != -1:
                    ret, thresh_img = cv2.threshold(
                        symb_img.img, otsus, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    symb_img.img = thresh_img
                if laplacian:
                    lap_img = cv2.Laplacian(symb_img.img, cv2.CV_64F)
                    symb_img.img = lap_img
                if canny != -1:
                    canny_img = cv2.Canny(symb_img.img, canny[0], canny[1])
                    symb_img.img = canny_img
                if rescale_global_mean:
                    scaled_img = symb_img.img / 255.0
                    symb_img.img = scaled_img - np.mean(scaled_img)
                if resize != -1:
                    old_size = symb_img.img.shape[:2]

                    delta_w = max(old_size) - old_size[1]
                    delta_h = max(old_size) - old_size[0]
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)

                    color = [0, 0, 0]
                    symb_img.img = cv2.copyMakeBorder(symb_img.img,
                                                      top,
                                                      bottom,
                                                      left,
                                                      right,
                                                      cv2.BORDER_CONSTANT,
                                                      value=color)

                    symb_img.img = cv2.resize(symb_img.img, (resize, resize))
