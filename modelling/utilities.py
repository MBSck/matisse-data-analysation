#!/usr/bin/env python3

__author__ = "Marten Scheuck"

import numpy as np
import matplotlib.pyplot as plt                 # Imports the matplotlib module for image processing
import time

from abc import ABCMeta, abstractmethod         # Import abstract class functionality
from typing import Union
from astropy.io import fits
from PIL import Image                           # Import PILLOW for image processing
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

def delta_fct(x: Union[int, float], y: Union[int, float]) -> int:
    """Dirac Delta measure

    Parameters
    ----------
    x: int | float
    y: int | float

    Returns
    -------
    int
        1 if 'x == y' or 0 else
    """
    return 1 if x == y else 0

def set_size(size: int, step: int,  centre: bool = None) -> np.array:
    """
    Sets the size of the model and its centre

    Parameters
    ----------
    size: int
        Sets the size of the model image and implicitly the x-, y-axis
    step: int
        The step size of the np.arange, that defines the axes
    centre: bool
        A set centre of the object. Will be set automatically if default 'None' is kept

    Returns
    -------
    radius: np.array
        The radius of the object
    """
    x = np.arange(0, size, step)
    y = x[:, np.newaxis]

    if centre is None:
        x0 = y0 = size//2
    else:
        x0 = centre[0]
        y0 = centre[1]

    return np.sqrt((x-x0)**2 + (y-y0)**2).astype(int)

def set_uvcoords() -> np.array:
    """Sets the uv coords for visibility modelling

    Returns
    -------
    np.array
        Visibility axis
    """
    u = np.arange(-150, 150)
    v = u[:, np.newaxis]

    B = np.sqrt(u**2+v**2).astype(int)

    # Avoid ZeroDivisionError
    # B[B == 0] == 1

    return B

def temperature_gradient(radius: float, q: float, r_0: float, T_0: float):
    """Temperature gradient model determined by power-law distribution. 

    Parameters
    ----------
    radius: float
        The specified radius
    q: float
        The power-law index
    r_0: float
        The initial radius
    T_0: float
        The temperature at r_0

    Returns
    -------
    temperature: float
        The temperature at a certain radius
    """
    try:
        power_factor = (radius/r_0)**q
    except ZeroDivisionError:
        power_factor = radius**q

    # Remove the ZeroDivisionError -> Bodge
    # TODO: Think of better way
    #  power_factor[power_factor == 0] = 1.

    return T_0/power_factor

def blackbody_spec(radius: float, q: float, r_0: float, T_0: float, wavelength: float):
    """Gets the blackbody spectrum at a certain T(r). Per Ring wavelength and temperature dependent

    Parameters
    ----------
    radius: float
        The predetermined radius
    q: float
        The power-law index
    r_0: float
        The initial radius
    T_0: float
        The temperature at r_0

    Returns
    -------
    Planck's law B_lambda(lambda, T): float
        The spectral radiance (the power per unit solid angle) of a black-body
    """
    T = temperature_gradient(radius, q, r_0, T_0)
    factor = (2*PLANCK_CONST*SPEED_OF_LIGHT**2)/wavelength**5

    exp_nominator = PLANCK_CONST*SPEED_OF_LIGHT
    exp_divisor = wavelength*BOLTZMAN_CONST*T
    exponent = np.exp(exp_nominator/exp_divisor)-1

    return factor/exponent


def do_plot(input_model, *args, mod: bool = False, vis: bool = False, both: bool = False) -> None:
    """Simple plot function for the models

    Parameters
    ----------
    args
        Different model inputs

    Returns
    -------
    None
    """
    # TODO: Make this take any number of arguments as well as any number of models
    model = input_model()

    if mod:
        plt.imshow(model.eval_model(*args))
    if vis:
        plt.imshow(model.eval_vis(*args), extent=(-150, 150, -150, 150))
    if both:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(model.eval_model(*args))
        ax2.imshow(model.eval_vis(*args), extent=(-150, 150, -150, 150))

        ax1.set_title("Model")
        ax1.set_xlabel("[px]")

        ax2.set_title("Vis")
        ax2.set_xlabel("u[m]")
        ax2.set_ylabel("v[m]")
    plt.show()


# Classes

class Model(metaclass=ABCMeta):
    """Abstract metaclass that initiates the models

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @abstractmethod
    def eval_model() -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass

    @abstractmethod
    def eval_vis() -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass


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

