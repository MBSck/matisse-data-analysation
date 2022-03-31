#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect
import time

from typing import Any, Dict, List, Union, Optional
from astropy.io import fits
from functools import wraps

from src.functionality.constant import *

# TODO: Check sr2mas conversion for pixel scaling
# TODO: Check linspace in set_size for accuracy? -> Don't use sampling

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

def m2au(length: Union[int, float]):
    """Converts units of [m] to [au]"""
    return length/AU_M

def orbit_au2arc(orbit_radius: Union[int, float],
                 distance: Union[int, float]):
    """Converts the orbital radius from [au] to [arc]

    Parameters
    ----------
    orbit_radius: int | float
        The radius of the star or its orbit
    distance: int | float
        The distance to the star

    Returns
    -------
    orbit: float
        The orbit in arcseconds
    """
    return orbit_radius/distance

def arc2rad(length_in_arc: Union[int, float]):
    """Converts the orbital radius from [arcsec] to [rad]"""
    return length_in_arc*ARCSEC2RADIANS

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

def rad2mas(angle: Optional[float] = None):
    """Converts from radians to milliarcseconds"""
    return 1/mas2rad(angle)

def sr2mas(pixel_size: float):
    """Converts the dimensions of an object from 'sr' to 'mas'. the result is
    in per pixel

    Parameters
    ----------
    pixel_size: float
        The size of one pixel in mas
    """
    return (pixel_size/(3600e3*180/np.pi))**2

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
    axis = np.roll(ax, (axis_size := len(ax))//2, 0)
    return np.diff(ax)[0]*wavelength/mas2rad()

def correspond_uv2scale(scaling: float, roll: int, uvcoords: np.ndarray) -> float:
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
    xcoord, ycoord = [roll+np.round(i[0]/scaling).astype(int) for i in uvcoords], \
            [roll+np.round(i[1]/scaling).astype(int) for i in uvcoords]
    return xcoord, ycoord

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

def set_size(size: int, sampling: Optional[int] = None,
             angles: List[float] = None) -> np.array:
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
    angles: List[float]
        A list of the three angles [ellipsis_angle, pos_angle inc_angle]

    Returns
    -------
    radius: np.array
        The radius
    xc: np.ndarray
        The x-axis used to calculate the radius
    """
    if (sampling is None):
        sampling = size

    x = mas2rad(np.linspace(-size//2, size//2, sampling))
    y = x[:, np.newaxis]

    if angles is not None:
        try:
            ellipsis_angle, pos_angle, inc_angle = angles
        except:
            raise RuntimeError(f"{inspect.stack()[0][3]}(): Check input"
                               " arguments, ellipsis_angles must be of the"
                               " form [ellipsis_angle, pos_angle, inc_angle]")

        a, b = x*np.sin(ellipsis_angle), y*np.cos(ellipsis_angle)
        ar, br = a*np.sin(pos_angle)+b*np.cos(pos_angle), \
                a*np.cos(pos_angle)-b*np.sin(pos_angle)

        return np.sqrt(ar**2+br**2*np.cos(inc_angle)**2), [ar, br]
    else:
        return np.sqrt(x**2+y**2), x

def zoom_array(array: np.ndarray, set_size: Optional[int] = None) -> np.ndarray :
    """Zooms in on an image by cutting of the zero-padding

    Parameters
    ----------
    array: np.ndarray
        The image to be zoomed in on
    set_size: int, optional
        The size for the image cut-off

    Returns
    -------
    np.ndarray
        The zoomed in array, with the zero-padding cut-off
    """
    array_center = len(array)//2
    if set_size is None:
        set_size = int(len(array)*0.15)

    ind_low, ind_high = array_center-set_size,\
            array_center+set_size

    return array[ind_low:ind_high, ind_low:ind_high]

def set_uvcoords(sampling: int, wavelength: float, angles: List[float] = None,
                 uvcoords: np.ndarray = None) -> np.array:
    """Sets the uv coords for visibility modelling

    Parameters
    ----------
    sampling: int | float
        The sampling of the (u,v)-plane
    wavelength: float
        The wavelength the (u,v)-plane is sampled at
    angles: List[float]
        A list of the three angles [ellipsis_angle, pos_angle inc_angle]
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

    else:
        axis = uvcoords
        u, v = np.array([i[0] for i in uvcoords]), \
                np.array([i[1] for i in uvcoords])

    if angles is not None:
        try:
            ellipsis_angle, pos_angle, inc_angle = angles
        except:
            raise RuntimeError(f"{inspect.stack()[0][3]}(): Check input"
                               " arguments, ellipsis_angles must be of the form"
                               " [ellipsis_angle, pos_angle, inc_angle]")

        # The ellipsis of the projected baselines
        a, b = u*np.sin(ellipsis_angle), v*np.cos(ellipsis_angle)

        # Projected baselines with the rotation by the positional angle of the disk semi-major axis theta
        ath, bth = a*np.sin(pos_angle)+b*np.cos(pos_angle), \
                a*np.cos(pos_angle)-b*np.sin(pos_angle)
        B = np.sqrt(ath**2+bth**2*np.cos(inc_angle)**2)
    else:
        B = np.sqrt(u**2+v**2)/wavelength

    return B, uvcoords

def sublimation_radius(T_sub: int, L_star: int, distance: float):
    """Calculates the sublimation radius of the disk

    Parameters
    ----------
    T_sub: int
        The sublimation temperature of the disk. Usually fixed to 1500 K
    L_star: int
        The star's luminosity in units of nominal solar luminosity
    distance: int
        Distance in parsec

    Returns
    -------
    R_sub: int
        The sublimation_radius in radians
    """
    L_star *= SOLAR_LUMINOSITY
    sub_radius_m = np.sqrt(L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*T_sub**4))
    sub_radius_au = m2au(sub_radius_m)
    sub_radius_arc = orbit_au2arc(sub_radius_au, distance)
    return arc2rad(sub_radius_arc)

def temperature_gradient(radius: float, r_0: Union[int, float],
                         q: float, T_0: int) -> Union[float, np.ndarray]:
    """Temperature gradient model determined by power-law distribution.

    Parameters
    ----------
    radius: float
        The specified radius
    r_0: float
        The initial radius
    q: float
        The power-law index
    T_0: float
        The temperature at r_0

    Returns
    -------
    temperature: float | np.ndarray
        The temperature at a certain radius
    """
    # q is 0.5 for flared irradiated disks and 0.75 for standard viscuous disks
    return T_0*(radius/r_0)**(-q)

def plancks_law_nu(radius: float, q: float,
                   r_0: Union[int, float], T_0: int,
                   wavelength: float) -> [float, np.ndarray]:
    """Gets the blackbody spectrum at a certain T(r). Wavelength and
    temperature dependent. The wavelength will be converted to frequency

    Parameters
    ----------
    radius: float
        The predetermined radius
    r_0: float
        The initial radius
    q: float
        The power-law index
    T_0: float
        The temperature at r_0
    wavelength: float
        The wavelength to be converted to frequency

    Returns
    -------
    planck's law/B_nu(nu, T): float | np.ndarray
        The spectral radiance (the power per unit solid angle) of a black-body
        in terms of frequency
    """
    T = temperature_gradient(radius, r_0, q, T_0)

    nu = SPEED_OF_LIGHT/wavelength
    factor = (2*PLANCK_CONST*nu**3)/SPEED_OF_LIGHT**2
    exponent = (PLANCK_CONST*nu)/(BOLTZMAN_CONST*T)

    return factor*(1/(np.exp(exponent)-1))

    def do_fit():
        """Does automatic gauss fits"""
        # Fits the data
        scaling_rad2arc = 206265

        # Gaussian fit
        fwhm = 1/scaling_rad2arc/1000           # radians

        # np.linspace(np.min(spat_freq), np.max(spat_freq), 25)
        xvals, yvals = self.baseline_distances, self.mean_bin_vis2
        pars, cov = curve_fit(f=gaussian, xdata=xvals, ydata=yvals, p0=[fwhm], bounds=(-np.inf, np.inf))
        # xvals = np.linspace(50, 150)/3.6e-6
        # fitted_model= np.square(gaussian(xvals, fwhm))
        ax.plot(xvals, gaussian(xvals, pars), label='Gaussian %.1f"'%(fwhm*scaling_rad2arc*1000))

        # Airy-disk fit
        fwhm = 3/scaling_rad2arc/1000           # radians
        fitted_model = np.square(airy(xvals, fwhm))
        ax.plot(xvals/1e6, fitted_model*0.15, label='Airy Disk %.1f"'%(fwhm*scaling_rad2arc*1000))
        ax.set_ylim([0, 0.175])
        ax.legend(loc='best')


if __name__ == "__main__":
    # get_px_scaling([i for i in range(0, 10)], 1e-5)
    print(sub_r := sublimation_radius(1500, 19, 140))
    radius, axis = set_size(size:=512)
    radius[radius < sub_r] = 0.
    print(temp:=temperature_gradient(radius, sub_r, 0.55, 1500))
    plt.imshow(temp)
    plt.show()
    print(temp[size//2, size//2])
    # plt.imshow(temp)
    # plt.show()

    # print(radius)
    # print(spectral_rad := plancks_law_nu(radius, 0.55, sub_r, 1500, 8e-6))
    # print(flux_jy := sr2mas(1.3)*spectral_rad*1e26)
    # print(flux_jy[size//2][size//2])
    # plt.imshow(flux_jy)
    # plt.show()
    ...

