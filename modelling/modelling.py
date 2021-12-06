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
from typing import Union

from constant import SPEED_OF_LIGHT, PLANCK_CONST, BOLTZMAN_CONST   # Import constants
from utilities import timeit

# TODO: Work data like RA, DEC, and flux into the script
# TODO: Use the delta functions to integrate up to a disk
# TODO: Check all analytical visibilities

# Set for debugging
# np.set_printoptions(threshold=sys.maxsize)

# Functions

def delta_fct(x: Union[int,  float], y: Union[int, float]) -> int:
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
    u = np.arange(-150, 150, 1)
    v = u[:, np.newaxis]
    return np.sqrt(u**2+v**2).astype(int)

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
    # power_factor[power_factor == 0] = 1.

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
        plt.imshow(model.eval_vis(*args))

    if both:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(model.eval_model(*args))
        ax2.imshow(model.eval_vis(*args))

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


class Delta(Model):
    """Delta function/Point source model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def eval_model(self, size: int, step: int = 1, flux: float = 1.) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.array([[0. for j in range(size)] if not i == size//2 else [0. if not j == size//2 else 1.*flux for j in range(size)] for i in range(size)])

    def eval_vis(self, size: int, flux: float = 1.) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.ones((size, size))*flux


class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
    size: int
        The size of the array that defines x, y-axis and constitutes the radius
    major: int
        The major determines the radius/cutoff of the model
    step: int
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    centre
        The centre of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def eval_model(self, size: int, major: int, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)
        return (flux/np.sqrt(np.pi/(4*np.log(2)*major)))*(np.exp(-4*np.log(2)*(radius**2)/(major**2)))

    def eval_vis(self, major: int, flux: float = 1.) -> np.array:
        # TODO: Somehow relate the visibilites to the real actual model analytically
        # TODO: This is not completely correct as well
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return np.exp(-(np.pi*major*B)**2/(4*np.log(2)))


class Ring(Model):
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
    centre
        The centre of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def eval_model(self, size: int, major: int, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)

        try:
            return np.array([[(flux*delta_fct(j, major/2))/(np.pi*major) for j in i] for i in radius])
        except ZeroDivisionError:
            return np.array([[(flux*delta_fct(j, major/2))/(np.pi) for j in i] for i in radius])

    def eval_vis(self, major: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return j0(2*np.pi*major*B)


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

    def eval_model(self, size: int, major: int, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)

        return np.array([[4*flux/(np.pi*(major**2)) if j <= major//2 else 0 for j in i] for i in radius])

    def eval_vis(self, major: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return 2*j1(np.pi*major*B)


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
    def eval_model(self, size: int, major: int, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)

        return np.array([[(6*flux/(np.pi*(major**2)))*np.sqrt(1-(2*j/major)**2) if j <= major//2 else 0 for j in i] for i in radius])

    def eval_vis(self, major: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return (3/(np.pi*major*B)**3)*(np.sin(np.pi*major*B)-np.pi*major*B*np.cos(np.pi*major*B))


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
    centre: bool
        The centre of the model, will be automatically set if not determined
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
    def eval_model(self, size: int, q: float, r_0: int, T_0: int, wavelength: float, distance: int, inclination_angle: int, step: int = 1, centre: bool = None) -> np.array:
        """Evaluates the Model

        Parameters
        ----------
        size: int
        q: float
        r_0: float
        T_0: float
        wavelength: float
        distance: float
        inc_angle: float
        step: int
        centre: bool

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        # Temperature gradient should be like 0.66, 0.65 or sth
        radius = set_size(size, step, centre)
        flux = blackbody_spec(radius, q, r_0, T_0, wavelength)

        factor = (2*np.pi/distance)*np.cos(inclination_angle)

        try:
            return factor*radius*flux
        except ZeroDivisionError:
            return factor*flux

    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        self.B = set_uvcoords()
        self.Bu, self.Bv = self.B*np.sin(self.pos_angle_measure), self.B*np.cos(self.pos_angle_measure)     # Baselines projected according to their orientation
        self.Buth, self.Bvth = self.Bu*np.sin(self.pos_angle_major)+self.Bv*np.cos(self.pos_angle_major), \
                self.Bu*np.cos(self.pos_angle_major)-self.Bv*np.sin(self.pos_angle_major)                   # Baselines with the rotation by the positional angle of the disk semi-major axis theta taken into account 
        self.B_proj = np.sqrt(self.Buth**2+(self.Bvth**2)*np.cos(self.inc_angle)**2)                        # Projected Baseline


        return (1/self.eval_model(0))*self.blackbody_spec(radius)


class IntegrateRings:
    """Adds 2D rings up/integrates them to create new models (e.g., a uniform disk or a ring structure)

    ...

    Methods
    -------
    add_rings2D():
        This adds the rings up to various models and shapes
    uniform_disk():
        Calls the add_rings() function with the right parameters to create a uniform disk
    disk():
        Calls the add_rings() function with the right parameters to create a disk with an inner rim
    """
    def __init__(self, size_model: int) -> None:
        self.size = size_model

    def add_rings(self, min_radius: int, max_radius: int, step_size: int, q: float, T_0: int, wavelength: float) -> None:
        """This adds the rings up to various models

        Parameters
        ----------
        min_radius: int
        max_radius: int
        step_size: int
        q: float
        T_0: int
        wavelength: float

        Returns
        -------
        None
        """
        # TODO: Make this more performant -> Super slow
        output_lst = np.zeros((self.size, self.size))

        for i in range(min_radius+1, max_radius+2, step_size):
            flux = blackbody_spec(i, q, min_radius, T_0, wavelength)
            ring_array = Ring().eval_model(self.size, i)
            output_lst[np.where(ring_array > 0)] = flux/(np.pi*max_radius)

        return output_lst

    @timeit
    def uniform_disk(self, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(0, radius, step_size, q, T_0, wavelength)

    @timeit
    def disk(self, inner_radius: int, outer_radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a disk with a inner ring

        See also
        --------
        add_rings()
        """
        return self.add_rings(inner_radius, outer_radius, step_size, q, T_0, wavelength)


if __name__ == "__main__":
    # for i in range(0, 100, 10):
    #    do_plot(InclinedDisk, 1024, .55, i, 6000, 8e-06, 1, 0, mod=True)
    # do_plot(InclinedDisk2D, 500, vis=True)
    integrate = IntegrateRings(500)
    integrate.uniform_disk(50)
    integrate.disk(20, 50)

