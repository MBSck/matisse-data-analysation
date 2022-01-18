#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union, Optional
from astropy.io import fits
from functools import wraps

from src.functionality.constant import *

# Functions

def timeit(func):
    """Simple timer decorator for functions"""
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
        An x value
    y: int | float
        An y value

    Returns
    -------
    int
        1 if 'x == y' or 0 else
    """
    return 1 if x == y else 0

def compare_arrays(array1: np.array, array2: np.array,
                   rtol: Union[int, float] = 1e-05, atol: Union[int, float] = 1e-08) -> [np.array, np.array]:
    """Compares two arrays for their differences in value and also if they are equal in a certain tolerance

    Parameters
    ----------
    array1: np.array
    array2: np.array
    rtol: int | float
        relative tolerance of difference
    atol: int | float
        absolute tolerance of difference

    Returns
    -------
    np.array
        The numerical difference between the arrays
    np.array
        An array of booleans that contain True/False if their difference is within a certain tolerance
    """
    return array1-array2, np.isclose(array1, array2, rtol=rtol, atol=atol)

def mas2rad(angle: Optional[Union[int, float]] = None):
    """Returns a given angle in mas/rad or the pertaining scaling factor"""
    if angle is None:
        return np.deg2rad(1/3.6e6)
    return np.deg2rad(angle/3.6e6)

def set_size(size: int, sampling: Optional[int] = None,  centre: Optional[int] = None, pos_angle: Optional[int] = None) -> np.array:
    """
    Sets the size of the model and its centre. Returns the polar coordinates

    Parameters
    ----------
    size: int
        Sets the size of the model image and implicitly the x-, y-axis.
        Size change for simple models functions like zero-padding
    sampling: int
        The sampling of the object-plane
    centre: bool
        A set centre of the object. Will be set automatically if default 'None' is kept
    pos_angle: int | None
        This sets the positional angle of the ellipsis if not None

    Returns
    -------
    radius: np.array
    """
    if (sampling is None) or (sampling < size):
        sampling = size

    x = np.linspace(0, size, sampling)
    y = x[:, np.newaxis]

    if centre is None:
        x0 = y0 = size//2
    else:
        x0 = centre[0]
        y0 = centre[1]

    xc, yc = (x-x0), (y-y0)

    if pos_angle is not None:
        pos_angle *= np.radians(1)
        xc, yc = xc*np.sin(pos_angle), yc*np.cos(pos_angle)

    return np.sqrt(xc**2+yc**2)*mas2rad()

def set_uvcoords(sampling: int, wavelength: float) -> np.array:
    """Sets the uv coords for visibility modelling

    Parameters
    ----------
    sampling: int | float
        The sampling of the (u,v)-plane
    wavelength: float
        The wavelength the (u,v)-plane is sampled at

    Returns
    -------
    baseline: np.array
    """
    if sampling < 300:
        sampling = 300

    # TODO: Fit the u,v sampling to the ft arrays
    B = np.linspace(-150, 150, sampling)

    # Star overhead sin(theta_0)=1
    u, v = B/wavelength, B[:, np.newaxis]/wavelength

    return np.sqrt(u**2+v**2)

def temperature_gradient(radius: float, q: float, r_0: Union[int, float], T_0: int):
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
    # q is 0.5 for flared irradiated disks and 0.75 for standard viscuous disks
    return T_0*(radius/r_0)**(-q)

def blackbody_spec(radius: float, q: float, r_0: Union[int, float], T_0: int, wavelength: float):
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


def do_plot(input_models: List[np.array], *args, ffft: bool = False, ft: Optional[np.array],
            mod: bool = False, vis: bool = False, both: bool = False) -> None:
    """Simple plot function for the models

    Parameters
    ----------
    input_model: np.arry
        Model input, in form of a 2D-np.array
    args
        Different model inputs
    fft: bool
        By default False, if toggled plots FFT instead of models
    ft: np.array | None
        The FFT to be plotted if fft is True
    mod: bool
        By default False, if toggled plots model
    vis: bool
        By default False, if toggled plots visibilities (FFT)
    both: bool
        By default False, if toggled plots both model and vis

    Returns
    -------
    None
    """
    # TODO: Make this take any number of arguments as well as any number of models
    for model in input_models:
        model = model()
        if fft:
            ...
        else:
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


if __name__ == "__main__":
    # readout = ReadoutFits("TARGET_CAL_INT_0001bcd_calibratedTEST.fits")

    # print(readout.get_uvcoords_vis2, "uvcoords")
    # readout.do_uv_plot(readout.get_uvcoords_vis2)
    # print(readout.get_ucoords(readout.get_uvcoords_vis2), readout.get_vcoords(readout.get_uvcoords_vis2))

    # radius= set_size(512, 1, None)
    # print(radius, "radius", theta, "theta")

    B = set_uvcoords(512, 8e-06)
    print(B)
    radius = set_size(512)
    r = set_size(512, 5)
    print(radius, radius.shape, "----", r, r.shape)

