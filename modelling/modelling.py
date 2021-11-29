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

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from scipy.special import j0, j1    # Import the Bessel function of 0th and 1st order

# TODO: Work data like RA, DEC, and flux into the script
# TODO: Use the delta functions to integrate up to a disk

# Set for debugging
# np.set_printoptions(threshold=sys.maxsize)

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

def set_uvcoords():
    """Sets the uv coords for visibility modelling"""
    u = np.arange(-150, 150, 1, float)
    v = u[:, np.newaxis]
    return np.sqrt(u**2+v**2)



def do_model_plot(*args):
    """Simple plot function for the models"""
    # TODO: Make plot function that displays all of the plots
    model = args[0](500, 200)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(model.eval_model())
    ax2.imshow(model.eval_vis2())
    plt.show()


# Classes 

class Delta:
    """Delta function/Point source model"""
    def __init__(self, size: float, step: float = 1., RA = None, DEC = None) -> None:
        self.size, self.step = size, step
        self.center = self.size//2
        self.RA, self.DEC = RA, DEC

        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return np.array([[0 for j in range(self.size)] if not i == self.center else [0 if not j == self.center else 1. for j in range(self.size)] for i in range(self.size)])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return np.ones((self.size, self.size))


class Gauss2D:
    """Two dimensional Gauss model, FFT is also Gauss"""
    def __init__(self, size: float, fwhm: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        self.size, self.step, self.center = size, step, center
        self.fwhm = fwhm
        self.flux = flux
        self.RA, self.DEC = RA, DEC

        self.r = set_size(size, fwhm, step, center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return (self.flux/np.sqrt(np.pi/(4*np.log(2)*self.fwhm)))*(np.exp(-4*np.log(2)*self.r**2/self.fwhm**2))

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return np.exp(-(np.pi*self.fwhm*self.B)**2/(4*np.log(2)))


class Ring2D:
    """Infinitesimal thin ring model"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        self.size, self.step, self.center = size, step, center
        self.major = major
        self.flux = flux
        self.RA, self.DEC = RA, DEC

        self.r = set_size(size, major, step, center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        # TODO: Fix problem that ring is not complete!
        return np.array([[(self.flux/(np.pi*self.major))*delta_fct(j, self.major/2) for j in i] for i in self.r])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return j0(2*np.pi*self.major*self.B)


class UniformDisk:
    """Uniformly bright disc"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        self.size, self.step, self.center = size, step, center
        self.major = major
        self.flux = flux
        self.RA, self.DEC = RA, DEC

        self.r = set_size(size, major, step, center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return np.array([[4*self.flux/(np.pi*self.major**2) if j <= self.major/2 else 0 for j in i] for i in self.r])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return (2*j1(np.pi*self.major*self.B))*(np.pi*self.major*self.B)


class OpticallyThinSphere:
    """"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        self.size, self.step, self.center = size, step, center
        self.major = major
        self.flux = flux
        self.RA, self.DEC = RA, DEC

        self.r = set_size(size, major, step, center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return np.array([[(6*self.flux/(np.pi*self.major**2))*np.sqrt(1-(2*j/self.major)**2) if j <= self.major/2 else 0 for j in i] for i in self.r])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return (3/(np.pi*self.major*self.B)**3)*(np.sin(np.pi*self.major*self.B)-np.pi*self.major*self.B*np.cos(np.pi*self.major*self.B))



if __name__ == "__main__":
    do_model_plot(Ring2D)

