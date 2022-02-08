#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from typing import Any, Dict, List, Union, Optional
from astropy.io import fits
from functools import wraps

from src.functionality.constant import *

# TODO: Check how the indices are sorted, why do they change? They change even independent of the scaling
# TODO: Change euclidean distance to interpolation in order to get coordinates

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

# Functions

def trunc(values, decs=0):
    """Truncates the floating point decimals"""
    return np.trunc(values*10**decs)/(10**decs)

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

def mas2rad(angle: Optional[Union[int, float, np.ndarray]] = None):
    """Returns a given angle in mas/rad or the pertaining scaling factor

    Parameters
    ----------
    angle: int | float | np.ndarray, optional
        The input angle(s)

    Returns
    -------
    float
        The angle in radians
    """
    if angle is None:
        return np.radians(1/3.6e6)
    return np.radians(angle/3.6e6)

def get_px_scaling(ax: np.ndarray, wavelength: float) -> float:
    """Gets the model's scaling from its sampling rate/size the wavelength and
    the array's dimensionalities into 1/radians.

    Parameters
    ----------
    ax: np.ndarray
        The axis from which the scaling is to be computed. For FFT it is
        np.fft.fftfreq and for analytical model, set_uvcoords

    Return
    ------
    float
        The px to meter scaling
    """
    axis_size = len(ax)
    roll = np.floor(axis_size/2).astype(int)
    axis = np.roll(ax, roll, axis=0)
    return (np.diff(ax)[0]/(np.deg2rad(1))*wavelength*axis_size)

def correspond_uv2scale(scaling: float, uvcoords: np.ndarray) -> float:
    """Calculates the axis scaling from the axis of an input image/model and
    returns it in meters baseline per pixel. Returns the uv-coords

    Parameters
    ----------
    scaling: float
        The scaling factor the uv-coords should be corresponded to
    uvcoords: np.array
        The real uvcoords

    Returns
    -------
    uvcoords: np.ndarray
        The rescaled uv-coords
    """
    return uvcoords/scaling

def get_distance(axis: np.ndarray, uvcoords: np.ndarray) -> np.ndarray:
    """Calculates the norm for a point. Takes the freq and checks both the
    u- and v-coords against it.
    Works only for models/image that have the same length in both dimensions.

    The indices of the output list evaluate to the indices of the input list

    Parameters
    ----------
    axis: np.ndarray
        The axis of which the distance is to be calculated
    uvcoords: np.ndarray
        The uvcoords that should be cross referenced

    Returns
    -------
    np.ndarray
        The indices of the closest matching points
    """
    # This makes a list of all the distances in the shape (size_img_array, uv_coords)
    distance_lst = [[np.sqrt((j-i)**2) for j in axis] for i in uvcoords]

    # This gets the indices of the elements closest to the uv-coords
    # TODO: Maybe make more efficient with numpy instead of list comprehension
    indices_lst = [[j for j, o in enumerate(i) if o == np.min(np.array(i))] for i in distance_lst]
    return np.ndarray.flatten(np.array(indices_lst))

def interpolate():
    ...

def correspond_uv2model(model_vis: np.ndarray, model_axis: np.ndarray,uvcoords: np.array,
                        dis: bool = False, intpol: bool = False) -> List:
    """This gets the indicies of rescaled given uvcoords to a image/model with
    either euclidean distance or interpolation and returns their vis2 values

    Parameters
    ----------
    uvcoords: np.array
        The to the image's size rescaled uvcoords
    dis: bool
        If enabled, corresponds via euclidean distance
    intpol: bool
        If enable, corresponds via interpolationg

    Returns
    -------
    np.array
        Values for the vis2
    """
    if dis:
        u_ind, v_ind = get_distance(model_axis, [i[0] for i in uvcoords]), \
                get_distance(model_axis, [i[1] for i in uvcoords])
        uv_ind =  zip(u_ind, v_ind)
    if intpol:
        ...

    return [model_vis[i[0], i[1]] for i in uv_ind], list(uv_ind)

def set_size(size: int, sampling: Optional[int] = None, centre: Optional[int] = None, pos_angle: Optional[int] = None) -> np.array:
    """
    Sets the size of the model and its centre. Returns the polar coordinates

    Parameters
    ----------
    size: int
        Sets the size of the model image and implicitly the x-, y-axis.
        Size change for simple models functions like zero-padding
    sampling: int, optional
        The sampling of the object-plane
    centre: bool, optional
        A set centre of the object. Will be set automatically if default 'None' is kept
    pos_angle: int | None, optional
        This sets the positional angle of the ellipsis if not None

    Returns
    -------
    radius: np.array
        The radius
    xc: np.ndarray
        The x-axis used to calculate the radius
    """
    if (sampling is None):
        sampling = size

    x = np.linspace(0, size, sampling)
    y = x[:, np.newaxis]

    if centre is None:
        x0 = y0 = size//2
    else:
        x0, y0 = centre

    xc, yc = x-x0, y-y0

    if pos_angle is not None:
        pos_angle *= np.radians(1)
        xc, yc = xc*np.sin(pos_angle), yc*np.cos(pos_angle)

    return np.sqrt(xc**2+yc**2)*mas2rad(), xc

def set_uvcoords(sampling: int, wavelength: float, uvcoords:
                 Optional[List[float]] = None) -> np.array:
    """Sets the uv coords for visibility modelling

    Parameters
    ----------
    sampling: int | float
        The sampling of the (u,v)-plane
    wavelength: float
        The wavelength the (u,v)-plane is sampled at
    uvcoords: List[float], optional
        If uv-coords are given, then the visibilities are calculated for

    Returns
    -------
    baselines: ArrayLike
        The baselines for the uvcoords
    B: ArrayLike
        The axis used to calculate the baslines
    """
    if uvcoords is None:
        axis = np.linspace(-150, 150, sampling)

        # Star overhead sin(theta_0)=1 position
        u, v = axis, axis[:, np.newaxis]

        B = np.sqrt(u**2+v**2)/wavelength
    else:
        u, v = np.array([i[0] for i in uvcoords]), \
                np.array([i[1] for i in uvcoords])

        B, axis = np.sqrt(u**2+v**2)/wavelength, uvcoords

    return B, axis

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
    numerator = (2*PLANCK_CONST*SPEED_OF_LIGHT**2)

    exp_numerator = PLANCK_CONST*SPEED_OF_LIGHT
    exp_divisor = wavelength*BOLTZMAN_CONST*T
    divisor = wavelength**5*np.exp(exp_numerator/exp_divisor)

    return numerator/divisor


if __name__ == "__main__":
    print(set_uvcoords(10, 8e-06, [[10, 8], [5, 10]]))
    print(np.exp(1e-9*1e-6))
    # readout = ReadoutFits("TARGET_CAL_INT_0001bcd_calibratedTEST.fits")

    # print(readout.get_uvcoords_vis2, "uvcoords")
    # readout.do_uv_plot(readout.get_uvcoords_vis2)
    # print(readout.get_ucoords(readout.get_uvcoords_vis2), readout.get_vcoords(readout.get_uvcoords_vis2))

    # radius= set_size(512, 1, None)
    # print(radius, "radius", theta, "theta")

    # B = set_uvcoords(512, 8e-06)
    # print(B)
    # radius = set_size(512)
    # r = set_size(512, 5)
    # print(radius, radius.shape, "----", r, r.shape)
