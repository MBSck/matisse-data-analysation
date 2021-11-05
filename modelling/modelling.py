#!/usr/bin/env python3

__author__ = "Marten Scheuck"

# Credit
# https://docs.astropy.org/en/stable/modeling/new-model.html#a-step-by-step-definition-of-a-1-d-gaussian-model
# - Implementation of a custom model with astropy
# https://docs.astropy.org/en/stable/modeling/index.html - Modelling with astropy
# https://stackoverflow.com/questions/62876386/how-to-produce-lorentzian-2d-sources-in-python - 2D Lorentz
# https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html - 2D Gaussian

import numpy as np
import matplotlib.pyplot as plt

# TODO: Rework the gauss fit and make it more generally applicable

# Functions


def gauss2d(x_0, y_0, mx=0, my=0, sx=1, sy=1):
    """2D gauss model"""
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x_0 - mx)**2. / (2. * sx**2.) + (y_0 - my)**2. / (2. * sy**2.)))


def uniform_disk(x_0, y_0, radius):
    """2D Uniform Disk model"""
    return np.sqrt(x_0**2+y_0**2) < radius


def ring2d(x_0, y_0, inner_radius, outer_radius):
    """2D Ring model"""
    return (np.sqrt(x_0**2+y_0**2) < outer_radius) & (inner_radius < np.sqrt(x_0**2+y_0**2))


def model_generation(model, *args):
    """This takes some parameters and generates some fake data"""
    x = np.linspace(-5., 5., 300)
    y = np.linspace(-5., 5., 300)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D

    return model(x, y, *args)


if __name__ == "__main__":
    plt.imshow(model_generation(ring2d, 0.9, 1.))
    plt.show()
