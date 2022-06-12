#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect
import time

from typing import Any, Dict, List, Union, Optional, Callable
from astropy.io import fits
from functools import wraps

from src.functionality.constant import *

# TODO: Make progress bar into a decorator and also keep the time of the
# process and show the max time

# TODO: Finish the fit function, but maybe implement it in the plotter instead?

# Functions

def progress_bar(progress: int, total: int):
    """Displays a progress bar

    Parameters
    ----------
    progress: int
        Total progress
    total: int
        Total iterations
    """
    percent = 100 * (progress/total)
    bar = '#' * int(percent) + '-' * (100-int(percent))
    print(f"\r|{bar}|{percent:.2f}% - {progress}/{total}", end='\r')

def trunc(values, decs=0):
    """Truncates the floating point decimals"""
    return np.trunc(values*10**decs)/(10**decs)

def chi_sq(data: np.ndarray, sigma: np.ndarray,
           model: np.ndarray) -> float:
    """The chi square minimisation"""
    return np.sum((data-model)**2/sigma)

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

def m2au(radius: Union[int, float]):
    """Converts units of [m] to [au]"""
    return radius/AU_M

def m2arc(radius: float, distance: int):
    """Converts [m] to [arcsec]"""
    return orbit_au2arc(m2au(radius), distance)

def m2mas(radius: float, distance: int):
    """Converts [m] to [mas]"""
    return m2arc(radius, distance)*1000

def m2rad(radius: float, distance: int):
    """Converts [m] to [rad]"""
    return arc2rad(m2arc(radius, distance))

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

def sr2mas(mas_size: float, sampling: int):
    """Converts the dimensions of an object from 'sr' to 'mas'. the result is
    in per pixel

    Parameters
    ----------
    mas_size: float
        The size of the image [mas]
    sampling: int
        The pixel sampling of the image
    """
    return (mas_size/(sampling*3600e3*180/np.pi))**2

def azimuthal_modulation(polar_angle: Union[float, np.ndarray],
                         modulation_angle: float,
                         amplitudes: List[List] = [[1, 1]],
                         order: Optional[int] = 1) -> Union[float, np.ndarray]:
    """Azimuthal modulation of an object

    Parameters
    ----------
    polar_angle: float | np.ndarray
        The polar angle of the x, y-coordinates
    amplitudes: List[List]
        The 'c' and 's' amplitudes
    order: int, optional
        The order of azimuthal modulation

    Returns
    -------
    azimuthal_modulation: float | np.ndarray
    """
    # TODO: Implement Modulation field like Jozsef?
    total_mod = 0
    for i in range(order):
        c, s = amplitudes[i]
        total_mod += (c*np.cos((i+1)*(polar_angle-modulation_angle)) + \
                      s*np.sin((i+1)*(polar_angle-modulation_angle)))

    modulation = np.array(1+total_mod)
    modulation[modulation < 0] = 0.
    return modulation

def set_size(mas_size: int, size: int, sampling: Optional[int] = None,
             incline_params: Optional[List[float]] = None) -> np.array:
    """Sets the size of the model and its centre. Returns the polar coordinates

    Parameters
    ----------
    mas_size: int
        Sets the size of the image [mas]
    size: int
        Sets the range of the model image and implicitly the x-, y-axis.
        Size change for simple models functions like zero-padding
    sampling: int, optional
        The pixel sampling
    incline_params: List[float], optional
        A list of the inclination parameters [axis_ratio, pos_angle] [mas, rad]

    Returns
    -------
    radius: np.array
        The radius
    xc: np.ndarray
        The x-axis used to calculate the radius
    """
    with np.errstate(divide='ignore'):
        fov_scale = mas_size/size

        if sampling is None:
            sampling = size

        x = np.linspace(-size//2, size//2, sampling, endpoint=False)*fov_scale
        y = x[:, np.newaxis]

        if incline_params:
            try:
                axis_ratio, pos_angle = incline_params[0], \
                        np.radians(incline_params[1])
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the"
                              " form [axis_ratio, pos_angle]")

            if axis_ratio < 1.:
                raise ValueError("The value of the axis_ratio cannot be < 1.0")

            xr, yr = -x*np.cos(pos_angle)+y*np.sin(pos_angle), \
                    (x*np.sin(pos_angle)+y*np.cos(pos_angle))/axis_ratio
            radius = np.sqrt(xr**2+yr**2)
            axis, phi = [xr, yr], np.arctan2(xr, yr)
        else:
            radius = np.sqrt(x**2+y**2)
            axis, phi = [x, y], np.arctan2(x, y)

        return radius, axis, phi

def zoom_array(array: np.ndarray, bounds: List) -> np.ndarray :
    """Zooms in on an image by cutting of the zero-padding

    Parameters
    ----------
    array: np.ndarray
        The image to be zoomed in on
    bounds: int
        The boundaries for the zoom, the minimum and maximum

    Returns
    -------
    np.ndarray
        The zoomed in array
    """
    min_ind, max_ind = bounds
    return array[min_ind:max_ind, min_ind:max_ind]

def set_uvcoords(wavelength: float, sampling: int, size: Optional[int] = 200,
                 angles: List[float] = None, uvcoords: np.ndarray = None) -> np.array:
    """Sets the uv coords for visibility modelling

    Parameters
    ----------
    wavelength: float
        The wavelength the (u,v)-plane is sampled at
    sampling: int
        The pixel sampling
    size: int, optional
        Sets the range of the (u,v)-plane in meters, with size being the
        longest baseline
    angles: List[float], optional
        A list of the three angles [ellipsis_angle, pos_angle inc_angle]
    uvcoords: List[float], optional
        If uv-coords are given, then the visibilities are calculated for

    Returns
    -------
    baselines: ArrayLike
        The baselines for the uvcoords
    uvcoords: ArrayLike
        The axis used to calculate the baselines
    """
    with np.errstate(divide='ignore'):
        if uvcoords is None:
            axis = np.linspace(-size, size, sampling)

            # Star overhead sin(theta_0)=1 position
            u, v = axis/wavelength, axis[:, np.newaxis]/wavelength

        else:
            axis = uvcoords/wavelength
            u, v = np.array([i[0] for i in uvcoords]), \
                    np.array([i[1] for i in uvcoords])

        if angles is not None:
            try:
                if len(angles) == 1:
                    pos_angle = angles
                    ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                            v*np.cos(pos_angle)-u*np.sin(pos_angle)
                    B = np.sqrt(ur**2+vr**2)*wavelength
                else:
                    axis_ratio, pos_angle, inc_angle = angles

                    ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                            (v*np.cos(pos_angle)-u*np.sin(pos_angle))/axis_ratio
                    B = np.sqrt(ur**2+vr**2*np.cos(inc_angle)**2)*wavelength

                axis = [ur, vr]
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the form"
                              " either [pos_angle] or "
                              " [ellipsis_angle, pos_angle, inc_angle]")

        else:
            B = np.sqrt(u**2+v**2)*wavelength
            axis = [u, v]

        return B, axis

def stellar_radius_pc(T_eff: int, L_star: int):
    """Calculates the stellar radius from its attributes and converts it from
    m to parsec

    Parameters
    ----------
    T_eff: int
        The star's effective temperature [K]
    L_star: int
        The star's luminosity [L_sun]

    Returns
    -------
    stellar_radius: float
        The star's radius [pc]
    """
    L_star *= SOLAR_LUMINOSITY
    stellar_radius_m = np.sqrt(L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*T_eff**4))
    return stellar_radius_m/PARSEC2M

def sublimation_temperature(r_sub: float, L_star: int, distance: int):
    """Calculates the sublimation temperature at the inner rim of the disk

    Parameters
    ----------
    r_sub: float
        The sublimation radius [mas]
    L_star: int
        The star's luminosity in units of nominal solar luminosity
    distance: int
        Distance in parsec

    Returns
    -------
    T_sub: float
        The sublimation temperature [K]
    """
    L_star *= SOLAR_LUMINOSITY
    r_sub /= m2mas(1, distance)
    return (L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*r_sub**2))**(1/4)

def sublimation_radius(T_sub: int, L_star: int, distance: int):
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
        The sublimation_radius [mas]
    """
    L_star *= SOLAR_LUMINOSITY
    sub_radius_m = np.sqrt(L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*T_sub**4))
    return m2mas(sub_radius_m, distance)

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
    with np.errstate(divide='ignore'):
        return T_0*(radius/r_0)**(-q)

def plancks_law_nu(T: Union[float, np.ndarray],
                   wavelength: float) -> [float, np.ndarray]:
    """Gets the blackbody spectrum at a certain T(r). Wavelength and
    dependent. The wavelength will be converted to frequency

    Parameters
    ----------
    T: float
        The temperature of the blackbody
    wavelength: float
        The wavelength to be converted to frequency

    Returns
    -------
    planck's law/B_nu(nu, T): float | np.ndarray
        The spectral radiance (the power per unit solid angle) of a black-body
        in terms of frequency
    """
    with np.errstate(divide='ignore'):
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
    radius, axis, phi = set_size(10, 2**6, 2**8)
