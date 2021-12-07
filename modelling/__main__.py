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

from scipy.special import j0, j1                                    # Import the Bessel function of 0th and 1st order

from constant import SPEED_OF_LIGHT, PLANCK_CONST, BOLTZMAN_CONST   # Import constants
from utilities import Model, timeit, temperature_gradient, \
        blackbody_spec, delta_fct, set_size, set_uvcoords, do_plot  # Import helper functions and baseclass for the models

# TODO: Work data like RA, DEC, and flux into the script
# TODO: Use the delta functions to integrate up to a disk
# TODO: Check all analytical visibilities

# Set for debugging
# np.set_printoptions(threshold=sys.maxsize)


def main()
    ...


if __name__ == "__main__":
    # do_plot(Gauss2D, 150, vis=True)
    # do_plot(UniformDisk, 150, vis=True)
    # do_plot(OpticallyThinSphere, 150, vis=True)

    integrate = IntegrateRings(512)
    rimmed_disk_thick = integrate.rimmed_disk(20, 50, optically_thick=True)
    plt.imshow(rimmed_disk_thick)
    plt.show()

    rimmed_disk_thin = integrate.rimmed_disk(20, 50)
    plt.imshow(rimmed_disk_thin)
    plt.show()

    '''
    integrate = IntegrateRings(512)
    rimmed_disk_thick = integrate.rimmed_disk(20, 50, optically_thick=True)
    plt.imshow(rimmed_disk_thick)
    plt.savefig("Rimmed_disk_thick.png")

    rimmed_disk_thin = integrate.rimmed_disk(20, 50)
    plt.imshow(rimmed_disk_thin)
    plt.savefig("Rimmed_disk_thin.png")

    optically_thin_disk = integrate.optically_thin_disk(50)
    plt.imshow(optically_thin_disk)
    plt.savefig("Optically_thin_disk.png")

    optically_thick_disk = integrate.optically_thick_disk(50)
    plt.imshow(optically_thick_disk)
    plt.savefig("Optically_thick_disk.png")
    '''

