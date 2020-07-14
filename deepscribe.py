import cv2
import numpy as np
from matplotlib import pyplot as plt

from symbol_image import Symbol_Image
import argparse
from glob import glob, iglob
from disp_multiple_images import show_images


class DeepScribe:
    """
    Loads, transforms, and displays OCHRE dataset images.
    """
    
    @staticmethod
    def get_command_line_args():
        """
        Receives command line arguments specifying what images to use.
        
        Returns:
            args (argparse object): Object storing each argument with its corresponding command line input.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--symbol', help='symbol query')

        # no limit should be input as 'max'
        parser.add_argument('-l',
                            '--limit',
                            help='limit on number of image results',
                            default=100)
        args = parser.parse_args()
        return args

    # Maybe change symbol_dict to return value instead of parameter
    @staticmethod
    def load_images(symbol_dict):
        """
        Loads images in BGR based on command line arguments and stores them as Symbol_Image objects in a dictionary.
        
        Parameters:
            symbol_dict (dict): Dictionary to store loaded image data as symbol name : Symbol_Image object pairs.

        """
        
        args = DeepScribe.get_command_line_args()

        if args.symbol is None:
            symb_query = "*"
        else:
            symb_query = args.symbol
        
        query = "a_pfa/" + symb_query + "_*.jpg"
        count = 0

        for fn in iglob(query):
            # find second occurence of "_" which marks the start of the uuid
            separator_idx = fn.find("_", 6)
            extension_idx = fn.rfind(".jpg")
            name = fn[6:separator_idx]
            name = name.upper().strip(' »«')
            uuid = fn[separator_idx+1:extension_idx]

            # not using cv2.imread() in order to read unicode filenames
            img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8),
                               cv2.IMREAD_UNCHANGED)
            symb_img = Symbol_Image(name, uuid, img)

            if name in symbol_dict:
                symbol_dict[name].append(symb_img)
            else:
                symbol_dict[name] = [symb_img]
            count += 1

            if args.limit != 'max':
                if count >= args.limit:
                    break

    @staticmethod
    def display_images(symbol_dict, color=False):
        """
        Takes dictionary of loaded image data and displays all images in one matplotlib plot.
        
        Paramters:
            symbol_dict (dict): Dictionary of loaded image data.
            color (boolean): Displays images in color if True, otherwise displays in grayscale.
        """
        
        images = []

        for s in symbol_dict.values():
            for symb_img in s:
                images.append(symb_img.img)

        show_images(images, color=color)

    # global_thresh can be a threshold value or -1 for none
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
        """
        Takes dictionary of loaded image data, transforms all images according to the parameters, and saves them back in the dictionary.
        
        Paramters:
            symbol_dict (dict): Dictionary of loaded image data.
            gray (boolean): Converts images to grayscale if True, otherwise leaves them in color.
            gauss_filter (int): Kernel size of gaussian filter. If -1, do not apply filter.
            bilat_filter (list): List of length 3 of cv2.bilateralFilter parameters. If -1, do not apply filter.
            global_thresh (int): Global threshold value. If -1, do not apply threshold.
            adapt_thresh_mean (int): Block size of adaptive mean threshold. If -1, do not apply threshold.
            adapt_thresh_gauss (int): Block size of adaptive gaussian threshold. If -1, do not apply threshold.
            otsus (int): Otsu's binarization threshold value. If -1, do not apply threshold.
            laplacian (boolean): Applies Laplacian operator if True, otherwise does not apply operator.
            canny (list): List of length 2 of cv2.Canny parameters. If -1, do not apply canny edge detection.
            rescale_global_mean (boolean): Normalizes all image pixels and subtracts out the global mean pixel value if True.
            resize (int): Target width and height of image to be resized to. If -1, do not resize image.
        """
        
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
                # TODO: is normalizing before resizing correct?
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

    @staticmethod
    def count_symbols(printing=False):
        """
        Count the number of different symbols represented by the images loaded by the command line arguments.
        
        Parameters:
            printing (boolean): Print out the symbol dictionary of symbol name : frequency pairs for the loaded images.
            
        Returns:
            A list of different symbol names that were represented by the images loaded by the command line arguments.
        """
        
        symbol_dict = {}
        args = DeepScribe.get_command_line_args()

        if args.symbol is None:
            symb_query = "*"
        else:
            symb_query = args.symbol
        
        query = "a_pfa/" + symb_query + "_*.jpg"
        count = 0

        for fn in iglob(query):
            # find second occurence of "_" which marks the start of the uuid
            separator_idx = fn.find("_", 6)
            name = fn[6:separator_idx]
            name = name.upper().strip(' »«')

            if name in symbol_dict:
                symbol_dict[name] += 1
            else:
                symbol_dict[name] = 1
            count += 1

            if args.limit != 'max':
                if count >= args.limit:
                    break
        
        if printing:
            print(len(symbol_dict))
            for s in symbol_dict:
                print(s, ":", symbol_dict[s])
            
            total = 0
            for s in symbol_dict.values():
                total += s
            print(total)

        # return len(symbol_dict)
        return list(symbol_dict.keys())
