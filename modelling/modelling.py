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

from abc import ABCMeta, abstractmethod                             # Import abstract class functionality
from scipy.special import j0, j1                                    # Import the Bessel function of 0th and 1st order
from constant import PLANCK, SPEED_OF_LIGHT, BOLTZMAN               # Import constants

# TODO: Work data like RA, DEC, and flux into the script
# TODO: Use the delta functions to integrate up to a disk

# Set for debugging
# np.set_printoptions(threshold=sys.maxsize)

# Functions

def delta_fct(x: float, y: float):
    """Dirac Delta measure

    Parameters
    ----------
    x: float
    y: float

    Returns
    -------
    int
    """
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

    return np.sqrt((x-x0)**2 + (y-y0)**2).astype(int)

def set_uvcoords():
    """Sets the uv coords for visibility modelling

    Returns
    -------
    np.array
        Visibility axis
    """
    u = np.arange(-150, 150, 1, float)
    v = u[:, np.newaxis]
    return np.sqrt(u**2+v**2)

def do_plot(*args) -> None:
    """Simple plot function for the models

    Parameters
    ----------
    args
        Different model inputs

    Returns
    -------
    None
    """
    # TODO: Make plot function that displays all of the plots
    model = args[0](size=500, major=200)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(model.eval_model())
    ax2.imshow(model.eval_vis())
    plt.show()


# Classes 

class Model(metaclass=ABCMeta):
    """Abstract metaclass that initiates the models

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        self.size, self.step = size, step
        self.center = center
        self.major = major
        self.flux, self.RA, self.DEC = flux, RA, DEC

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


class Delta(Model):
    """Delta function/Point source model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float = None, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

    def eval_model(self) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.array([[0 for j in range(self.size)] if not i == self.center else [0 if not j == self.center else 1. for j in range(self.size)] for i in range(self.size)])

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.ones((self.size, self.size))


class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return (self.flux/np.sqrt(np.pi/(4*np.log(2)*self.major)))*(np.exp(-4*np.log(2)*self.r**2/self.major**2))

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.exp(-(np.pi*self.major*self.B)**2/(4*np.log(2)))


class Ring2D(Model):
    """Infinitesimal thin ring model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        try:
            return np.array([[(self.flux*delta_fct(j, self.major/2))/(np.pi*self.major) for j in i] for i in self.r])
        except ZeroDivisionError:
            return np.array([[(self.flux*delta_fct(j, self.major/2))/(np.pi*1) for j in i] for i in self.r])

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return j0(2*np.pi*self.major*self.B)


class UniformDisk(Model):
    """Uniformly bright disc model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.array([[4*self.flux/(np.pi*self.major**2) if j <= self.major/2 else 0 for j in i] for i in self.r])

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return (2*j1(np.pi*self.major*self.B))*(np.pi*self.major*self.B)


class OpticallyThinSphere(Model):
    """Optically Thin Sphere model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None) -> None:
        super().__init__(size, major, step, flux, RA, DEC, center)

        self.r = set_size(self.size, self.major, self.step, self.center)
        self.B = set_uvcoords()

    def eval_model(self) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.array([[(6*self.flux/(np.pi*self.major**2))*np.sqrt(1-(2*j/self.major)**2) if j <= self.major/2 else 0 for j in i] for i in self.r])

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return (3/(np.pi*self.major*self.B)**3)*(np.sin(np.pi*self.major*self.B)-np.pi*self.major*self.B*np.cos(np.pi*self.major*self.B))


class InclinedDisk(Model):
    """By a certain position angle inclined disk

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined
    r_0: float
        
    T_0: float
        
    q: float
        
    inc_angle: float
        
    pos_angle_major: float
        
    pos_angle_measurement: float
        
    wavelength: float
        
    distance: float
        

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, size: float, major: float, step: float = 1., flux: float = 1., RA = None, DEC = None, center = None, r_0: float = 1., T_0: float = 1., q: float = 1., \
                 inc_angle: float = None, pos_angle_major: float = None, pos_angle_measurement: float = None,  wavelength: float = 8e-6, distance: float = None) -> None:
        super().__init__(size, step, major, flux, RA, DEC, center)
        # TODO: Temperature gradient should be like 0.66, 0.65 or sth
        # Variables
        self.inc_angle = inc_angle                          # The inclination angle of the disk
        self.pos_angle_major = pos_angle_major              # The positional angle of the disk semi-major axis
        self.pos_angle_measure = pos_angle_measurement      # The positional angle of the measurement (B projected according to its orientation)
        self.r_0, self.T_0 = r_0, T_0                       # T_0 is temperature at r_0
        self.q = q                                          # Power law index, depends on the type of disk (flat or flared)

        # Calculated variables
        self.r = set_size(self.size, self.major, self.step, self.center)

        # Calculation of the projected Baseline
        self.B = set_uvcoords()
        self.Bu, self.Bv = self.B*np.sin(self.pos_angle_measure), self.B*np.cos(self.pos_angle_measure)     # Baselines projected according to their orientation
        self.Buth, self.Bvth = self.Bu*np.sin(self.pos_angle_major)+self.Bv*np.cos(self.angle_major), \
                self.Bu*np.cos(self.pos_angle_major)-self.Bv*np.sin(self.angle_major)                       # Baselines with the rotation by the positional angle of the disk semi-major axis theta taken into account 
        self.B_proj = np.sqrt(Buth**2+(Bvth**2)*np.cos(self.inc_angle)**2)                                  # Projected Baseline

    def temperature(self, radius: float):
        """Gets the temperature at a certain radius

        Parameters
        ----------
        radius: float
            The specified radius

        Returns
        -------
        temperature: float
            The temperature at a certain radius
        """
        return self.T_0*(radius/self.r_0)**(-self.q)

    def blackbody_spec(self, radius: float):
        """Gets the blackbody spectrum at a certain T(r). Per Ring wavelength and temperature dependent

        Parameters
        ----------
        radius: float
            The predetermined radius

        Returns
        -------
        blackbody_spec: float
            The blackbody spectrum at a certain radius and temperature
        """
        return ((2*PLANCK*SPEED_OF_LIGHT**2)/(self.wavelength**5)) * \
            (np.exp((PLANCK*SPEED_OF_LIGHT)/(self.wavelength*BOLTZMAN*self.temperature(radius)))-1)**(-1)

    def eval_model(self, radius: float, inc_angle: float, distance: float) -> np.array:
        """Evaluates the Model

        Parameters
        ----------
        radius: float
        inc_angle: float
        distance: float

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return (2*np.pi/distance)*np.cos(inc_angle)*radius*self.blackbody_spec(radius)

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return (1/self.eval_model(0))*self.blackbody_spec(radius)


class IntegrateRings:
    """Adds 2D rings up/integrates them to create new models (e.g., a uniform disk or a ring structure)

    ...

    """
    def __init__(self, size_model: int):
        self.size = size_model

    def add_rings2D(self, radius: int, max_radius: int, step_size: int):
        """This adds the rings up to various models"""
        # TODO: Make this more performant -> Super slow
        output_lst = np.zeros((self.size, self.size))

        for i in range(radius, max_radius+1, step_size):
            ring_array = Ring2D(500, i).eval_model()
            ind_x_array, ind_y_array = np.where(ring_array > 0)
            index_lst = [i for i in zip(list(ind_x_array), list(ind_y_array))]
            for i, o in index_lst:
                output_lst[i][o] = 1/(np.pi*i)

        plt.imshow(output_lst)
        plt.show()

    def uniform_disk(self, max_radius: int, step_size: int = 1):
        return self.add_rings2D(0, max_radius, step_size)

    def ring(self, radius: int, max_radius: int, step_size: int = 1, optically_thin: bool = True):
        return self.add_rings2D(radius, max_radius, step_size)


if __name__ == "__main__":
    # do_plot(Gauss2D)
    # integ = IntegrateRings(500).uniform_disk(50)
    integ = IntegrateRings(500).ring(30, 50)
    # do_model_plot(Ring2D)
