#!/usr/bin/env python3

__author__ = "Marten Scheuck"

import numpy as np
import matplotlib.pyplot as plt     # Imports the matplotlib module for image processing
import time

from astropy.io import fits
from PIL import Image               # Import PILLOW for image processing
from functools import wraps

# Functions

def timeit(func):
    @wraps(func)
    def timed_func(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        et = time.time()
        print(f"{func.__name__} execution took: {et-st} sec")
        return result
    return timed_func

# Classes

class ImageProcessing:
    """All functionality to process an image/model and use the uv-coords on it"""
    #TODO: Approach to uv-resizing; Max and Min value, resizing the coordinates. However no max and min value that corresponds?
    #TODO: Enlarge the image and the map the coordinates onto it, sizing is still weird?
    def __init__(self, path_to_img):
        self.path_to_img = path_to_img

    def read_image_into_nparray(self):
        """Checks the input if it is an np.array and if not reads it in as such"""
        if isinstance(self.path_to_img, np.ndarray):
            return self.path_to_img
 
        return plt.imread(self.path_to_img)

    @property
    def get_img_size(self):
        """Gets the size (width and height) of the image"""
        with Image.open(self.path_to_img) as img:
            return np.array(list(img.size))

class ReadoutFits:
    """All functionality to work with '.oifits/.fits'-files"""
    def __init__(self, fits_file):
        self.fits_file = fits_file

    def get_info(self):
        """Gets the header's info"""
        with fits.open(self.fits_file) as hdul:
            return hdul.info()

    def get_header(self, hdr):
        """Reads out the specified data"""
        return repr(fits.getheader(self.fits_file, hdr))

    def get_data(self, hdr, sub_hdr):
        """Gets a specific set of data from a header"""
        with fits.open(self.fits_file) as hdul:
            return (hdul[hdr].data)[sub_hdr]

    def get_column_names(self, hdr):
        """Fetches the columns of the header"""
        with fits.open(self.fits_file) as hdul:
            return (hdul[hdr].columns).names

    @property
    def get_uvcoords_vis2(self):
        """Fetches the u, v coord-lists and merges them as well as the individual components"""
        return np.array([i for i in zip(self.get_data(4, "ucoord"), self.get_data(4, "vcoord"))])

    @staticmethod
    def get_ucoords(uvcoords: np.array):
        """Splits a 2D-np.array into its 1D-components, in this case the u-coords"""
        return np.array([item[0] for item in uvcoords])

    @staticmethod
    def get_vcoords(uvcoords: np.array):
        """Splits a 2D-np.array into its 1D-components, in this case the v-coords"""
        return np.array([item[1] for item in uvcoords])


if __name__ == "__main__":
    readout = ReadoutFits("TARGET_CAL_INT_0001bcd_calibratedTEST.fits")
    img_proc = ImageProcessing("Michelson.png")

    print(readout.get_uvcoords_vis2, "uvcoords")
    # print(img_proc.get_img_size)
    readout.do_uv_plot(readout.get_uvcoords_vis2)
    print(readout.get_ucoords(readout.get_uvcoords_vis2), readout.get_vcoords(readout.get_uvcoords_vis2))

