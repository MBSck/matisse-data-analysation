#!/usr/bin/env python3

"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package"""

# TODO: Make it possible to fit uv-points to model data of vis models and not
# only FFT -> Make the rescaling function outside of the FFT?
# TODO: Make documentation in Evernote of this file and then remove unncessary
# comments
# TODO: Maybe make the MCMC fitter into class that takes care of all the
# model's parameters

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from src.models import Ring
from typing import Any, Dict, List, Union, Optional

def model(theta: List, wavelength: float):
    """The model defined for the MCMC process"""
    sampling, fwhm = theta
    model = Ring.eval_vis(sampling, fwhm, wavelength)
    return model

def lnlike(theta: np.array, u: float, v: float, verr: float):
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points. I.e. it is
    more important"""
    return -0.5*np.sum((y-)/yerr)

def lnprior(theta):
    """This function checks if all variables are within their priors (it is
    setting the same). If all priors are satisfied it needs to return '0.0' and
    if not then '-np.inf'"""
    sampling, fwhm = theta
    if sampling > 0  and sampling < 5500 and fwhm < 1000 and fwhm > 1:
        return 0.0
    else:
        return -np.inf

def lnprob(theta: np.array, x: float, y: float, yerr: float):
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
    return lp + lnlike(theta, x, y, yerr)

def main(p0, nwalkers, niter_burn, niter, ndim, lnprob, data) -> np.array:
    """"""
    # The EnsambleSampler gets the parameters. The args are the args put into the
    # lob_prob function. Additional parameter a can be used for the stepsize. None is
    # the default 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, a=6)

    # Burn-in of the sampler. Explores the parameter space. The walkers get settled
    # into the maximum of the density. Saves the walkers in the state variable
    print("Running burn-in...")
    p0 = sampler.run_mcmc(p0, niter_burn)

    # Resets the chain to remove burn in samples
    sampler.reset()

    # Do production. Starts from the final position of burn-in chain (rstate0 is
    # the state of the internal random number generator)
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return pos, prob, state

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

def plotter(sampler, wavelength: float) -> None:
    """Plot the samples to get estimate of the density that has been sampled, to
    test if sampling went well"""

    # This gets the parameter values for every walker at each step in the chain.
    # Array of shape (steps, nwalkers, ndim)
    samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    best_fit_model = model(theta_max)
    plt.imshow(best_fit_model)
    plt.show()

def plot_posterior_spread(sampler):
    """Plots the corner plot of the posterior spread"""
    labels = []
    samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
    fig = corner.corner(samples, show_titles=True, labels=labels,
                        plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])

if __name__ == "__main__":
    # Set the data to be fitted
    verr =  # error for uv is arbitrary, 2%?
    data = (u, v, verr)

    # Set the initial values for the parameters 
    initial = np.array([300, 10.])

    # The number of walkers (must be even) and the number of dimensions/parameters
    nwalkers, ndim = 250, len(initial)

    # Sets the steps of the burn-in and the max. steps
    niter_burn, niter = 100, 10000

    # This vector defines the starting points of each walker for the amount of
    # dimensions
    p0 = np.random.rand(nwalkers, ndim)

    # This calls the MCMC fitting
    main(p0, nwalkers, niter_burn, niter, ndim, lnprob, data)

    # This plots the resulting model
    plotter(sampler, 8e-06)
