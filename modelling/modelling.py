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

from abc import ABCMeta, abstractmethod     # Import abstract class functionality
from scipy.special import j0, j1            # Import the Bessel function of 0th and 1st order

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
    model = args[0](size=500, major=200)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(model.eval_model())
    ax2.imshow(model.eval_vis2())
    plt.show()


# Classes 

class Model(metaclass=ABCMeta):
    """Abstract metaclass that initiates the models"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        self.size, self.step = size, step
        self.center = center
        self.major = major
        self.flux, self.RA, self.DEC = flux, RA, DEC

    @abstractmethod
    def eval_model() -> np.array:
        """Evaluates the model"""
        pass

    @abstractmethod
    def eval_vis2() -> np.array:
        """Evaluates the visibilities of the model"""
        pass


class Delta(Model):
    """Delta function/Point source model"""
    def __init__(self, size: float, major: float = None, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return np.array([[0 for j in range(self.size)] if not i == self.center else [0 if not j == self.center else 1. for j in range(self.size)] for i in range(self.size)])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return np.ones((self.size, self.size))


class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return (self.flux/np.sqrt(np.pi/(4*np.log(2)*self.major)))*(np.exp(-4*np.log(2)*self.r**2/self.major**2))

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return np.exp(-(np.pi*self.major*self.B)**2/(4*np.log(2)))


class Ring2D(Model):
    """Infinitesimal thin ring model"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        # TODO: Fix problem that ring is not complete!
        return np.array([[(self.flux/(np.pi*self.major))*delta_fct(j, self.major//2) for j in i] for i in self.r])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return j0(2*np.pi*self.major*self.B)


class UniformDisk(Model):
    """Uniformly bright disc model"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return np.array([[4*self.flux/(np.pi*self.major**2) if j <= self.major/2 else 0 for j in i] for i in self.r])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return (2*j1(np.pi*self.major*self.B))*(np.pi*self.major*self.B)


class OpticallyThinSphere(Model):
    """Optically Thin Sphere model"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model"""
        return np.array([[(6*self.flux/(np.pi*self.major**2))*np.sqrt(1-(2*j/self.major)**2) if j <= self.major/2 else 0 for j in i] for i in self.r])

    def eval_vis2(self) -> np.array:
        """Evaluates the visibilities of the model"""
        return (3/(np.pi*self.major*self.B)**3)*(np.sin(np.pi*self.major*self.B)-np.pi*self.major*self.B*np.cos(np.pi*self.major*self.B))


class InclinedDisk(Model):
    """By a certain position angle inclined disk"""
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None, r_0: float = 1., T_0: float = 1., q: float = 1.,  pos_angle = None, wavlength: float = 8*10**(-6)) -> None:
        super().__init__(size, step, major, flux, RA, DEC, center)

        # Additional variables
        self.r_0, self.T_0 = r_0, T_0   # T_0 is temperature at r_0
        self.q = q                      # Power law index, depends on the type of disk (flat or flared)

        # Calculated variables
        self.r = set_size(self.size, self.major, self.step, self.center)
        self.temp = T_0*(r/r_0)**(-q)
        self.bbspec = ((2*h*c**2)/(wavelength**5))         # Blackbody spectrum per Ring, wavelength and temperature dependent
        self.B_uth, self.B_vth = 0, 0        # Baselines projected according to their orientation, B_{u, thetha}, B_{v, thetha}
        self.B = 0                        # Projected Baseline

    def eval_model(self) -> np.array:
        ...

    def eval_vis2(self) -> np.array:
        ...

class IntegrateRings():
    """Adds rings up to create new models"""
    # TODO: Base this on the inclined disk model
    def __init__(self):
        ...

    def uniform_disk():
        ...

if __name__ == "__main__":
    do_model_plot(Ring2D)
