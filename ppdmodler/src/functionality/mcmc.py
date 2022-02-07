#!/usr/bin/env python3

"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package"""

import matplotlib.pyplot as plt

from src.functionality.utilities import interpolate

# TODO: Make it possible to fit uv-points to model data of vis models and not
# only FFT -> Make the rescaling function outside of the FFT?
# TODO: Make documentation in Evernote of this file and then remove unncessary
# comments
# TODO: Maybe make the MCMC fitter into class that takes care of all the
# model's parameters

# TODO: Make the uv-coordinates into visibility so that the visibility can be
# directly compared to the one that is expected

# TODO: Make the code faster by only comparing six values

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.models import Gauss2D
from src.functionality.readout import ReadoutFits
from src.functionality.utilities import correspond_uv2model

def model(theta: np.ndarray, sampling: int, wavelength: float) -> np.ndarray:
    """The model defined for the MCMC process"""
    fwhm = theta
    model = Gauss2D()
    model_vis = model.eval_vis(sampling, fwhm, wavelength)
    model_axis = model.axis_vis
    return model_vis, model_axis

def lnlike(theta: np.ndarray, sampling: int, wavelength: float, vis2data: np.ndarray,
           vis2err: np.ndarray, uvcoords: np.ndarray):
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points. I.e. it is
    more important"""
    visdatamod, vis_axis = model(theta, sampling, wavelength)
    visdatamod, uv_ind  = correspond_uv2model(visdatamod, vis_axis, uvcoords, True)
    vis2datamod = visdatamod*np.conj(visdatamod)
    return -0.5*np.sum((vis2data-vis2datamod)**2/vis2err)

def lnprior(theta):
    """This function checks if all variables are within their priors (it is
    setting the same). If all priors are satisfied it needs to return '0.0' and
    if not then '-np.inf'"""
    fwhm = theta
    if fwhm < 100. and fwhm > 0.1:
        return 0.0
    else:
        return -np.inf

def lnprob(theta: np.ndarray, sampling: int, wavelength: float, vis2data: np.array,
           vis2err: np.ndarray, uvcoords: np.ndarray) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    theta: List
        A vector that contains all the parameters of the model

    Returns
    -------
    """
    lp = lnprior(theta)
    if not lp == 0.0:
        return -np.inf
    return lp + lnlike(theta, sampling, wavelength, vis2data, vis2err, uvcoords)

def main(p0, nwalkers, niter_burn, niter, ndim, lnprob, data) -> np.array:
    """"""
    # The EnsambleSampler gets the parameters. The args are the args put into the
    # lob_prob function. Additional parameter a can be used for the stepsize. None is
    # the default 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    # Burn-in of the sampler. Explores the parameter space. The walkers get settled
    # into the maximum of the density. Saves the walkers in the state variable
    print("Running burn-in...")
    p0 = sampler.run_mcmc(p0, niter_burn)

    # Resets the chain to remove burn in samples and sets the walkers lower
    sampler.reset()

    # Do production. Starts from the final position of burn-in chain (rstate0 is
    # the state of the internal random number generator)
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

def test_model(sampler) -> None:
    # Another test of whether or not the sampling went well is the mean acceptance
    # fraction and the integrated autocorrelation time
    # Acceptance fraction has an entry for each walker -> It is a vector with the
    # dimensions of the steps
    # As a rule of thumb if the acceptance_fraction is below 0.2, the a parameter
    # needs to be decreased, and if it is above 0.5 then the a parameter needs to
    # be increased
    acceptance = np.mean(sampler.acceptance_fraction)

    # The autocorrelation time is a vector with ndim dimensions
    autocorr= np.mean(sampler.get_autocorr_time())
    print(f"Mean acceptance fraction {acceptance} and the autcorrelation time {autocorr}")

def model_plot(sampler, sampling: int, wavelength: float) -> None:
    """Plot the samples to get estimate of the density that has been sampled, to
    test if sampling went well"""

    # This gets the parameter values for every walker at each step in the chain.
    # Array of shape (steps, nwalkers, ndim)
    samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    print(theta_max)
    best_fit_model = model(theta_max, sampling, wavelength)[0]
    plt.imshow(best_fit_model)
    plt.savefig("model_plot.png")

def corner_plot(samples):
    """Plots the corner plot of the posterior spread"""
    # samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
    fig = corner.corner(samples)
    plt.show()

if __name__ == "__main__":
    # Set the initial values for the parameters 
    fwhm = 50.
    initial = np.array([fwhm])

    # Sets the wavelength the model is to be evaluated at
    sampling, wavelength = 512, 10e-06

    # Readout Fits for real data
    f = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(f)
    vis2data, vis2err = readout.get_vis24wl(wavelength)
    uvcoords = readout.get_uvcoords()

    # Set the data to be fitted. Error arbitrary, set to 1%
    data = (sampling, wavelength, vis2data, vis2err, uvcoords)

    # The number of walkers (must be even) and the number of dimensions/parameters
    nwalkers, ndim = 5, len(initial)

    # Sets the steps of the burn-in and the max. steps
    niter_burn, niter = 50, 500

    # This vector defines the starting points of each walker for the amount of
    # dimensions
    p0 = np.random.rand(nwalkers, ndim)

    # This calls the MCMC fitting
    # sampler, pos, prob, state = main(p0, nwalkers, niter_burn, niter, ndim, lnprob, data)
    # with open("sampler.npy", "wb") as f:
        # np.save(f, sampler.flatchain)

    # This plots the resulting model
    # model_plot(sampler, sampling, wavelength)

    # This plots the corner plots of the posterior spread
    samples = np.load("sampler.npy")
    corner_plot(samples)
