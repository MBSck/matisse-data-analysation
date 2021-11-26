#!/usr/bin/env python3

__author__ = "Marten Scheuck"

# Credit
# https://docs.astropy.org/en/stable/modeling/new-model.html#a-step-by-step-definition-of-a-1-d-gaussian-model
# - Implementation of a custom model with astropy
# https://docs.astropy.org/en/stable/modeling/index.html - Modelling with astropy
# https://stackoverflow.com/questions/62876386/how-to-produce-lorentzian-2d-sources-in-python - 2D Lorentz
# https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html - 2D Gaussian

# Info
# For RA, DEC -> Set it at the centre and make pixel scaling conversion just like for fft, but in this case not for uv-coords
# Just multiply with the flux

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# TODO: Work data like RA, DEC, and flux into the script
# TODO: Use the delta functions to integrate up to a disk

# Functions

def delta_fct(x: float, y: float):
    """Dirac Delta measure"""
    return 1 if x == y else 0

def set_size(size: float, major: float, step: float,  center = None):
    """
    Sets the size of the model and its centre

    Parameters
    ----------
    size: float
        Sets the size of the model image and implicitly the x-, y-axis
    major: float
        Sets the constant radius of the model (e.g., fwhm for a gauss model)
    step: float
        The step size of the np.arange, that defines the axes

    Returns
    -------
    radius: np.array
        The radius of the object
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size//2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.sqrt((x-x0)**2 + (y-y0)**2)



def gauss2d(size: float, fwhm: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None):
    """2D symmetric gaussian model"""
    r = set_size(size, fwhm, step, center)

    return (flux/np.sqrt(np.pi/(4*np.log(2)*fwhm)))*(np.exp(-4*np.log(2)*r**2/fwhm**2))

def uniform_disk(size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None):
    """Uniformly bright disc"""
    r = set_size(size, major, step, center)

    return np.array([[4*flux/(np.pi*major**2) if j <= major/2 else 0 for j in i] for i in r])

def ring2d(size: float, major: float, step: float  = 1.,  flux: float = 1., RA = None, DEC = None, center = None):
    """Infinitesimal thin ring"""
    # TODO: Fix the problem that the ring is not complete?
    r = set_size(size, major, step, center)

    return np.array([[(flux/(np.pi*major))*delta_fct(j, major/2) for j in i] for i in r])

def optically_thin_sphere(size: float, major: float, step: float = 1, flux: float = 1., RA = None, DEC = None, center = None):
    """Optically thin sphere"""
    r = set_size(size, major, step, center)

    return np.array([[(flux*6/(np.pi*major**2))*np.sqrt(1-(2*j/major)**2) if j <= major/2 else 0 for j in i] for i in r])

def do_model_plot(*args):
    """Simple plot function for the models"""
    # TODO: Make plot function that displays all of the plots
    plt.imshow(args[0](500, 200))
    plt.show()

if __name__ == "__main__":
    do_model_plot(optically_thin_sphere)

