#!/usr/bin/env python3

"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.models import Gauss2D
from src.functionality.fourier import FFT
from src.functionality.readout import ReadoutFits
from src.functionality.utilities import trunc

# TODO: Think of way to implement the FFT fits -> See Jacob's code
# TODO: Make documentation in Evernote of this file and then remove unncessary
# comments
# TODO: Make this more generally applicable
# TODO: Make plots of the model + fitting, that show the visibility curve, and
# two more that show the # fit of the visibilities and the closure phases to the measured one

class MCMC:
    def __init__(self, model, mc_params: List[float], data: List, numerical:
                 bool = True) -> None:
        self.model = model()
        self.data = data
        self.p0, self.nw, self.nd, self.nib, self.ni = mc_params
        self.numerical = numerical

    def pipeline(self) -> None:
        sampler, pos, prob, state = self.do_fit()

        # This plots the corner plots of the posterior spread
        mcmc.plot_corner(sampler)

        # This plots the resulting model
        # mcmc.plot_model_and_vis_curve(sampler, sampling, wavelength)

    def do_fit(self) -> np.array:
        """The main pipline that executes the combined mcmc code and fits the
        model"""
        # The EnsambleSampler gets the parameters. The args are the args put into the
        # lob_prob function. Additional parameter a can be used for the stepsize. None is
        # the default 
        sampler = emcee.EnsembleSampler(self.nw, self.nd, self.lnprob, args=self.data)

        # Burn-in of the sampler. Explores the parameter space. The walkers get settled
        # into the maximum of the density. Saves the walkers in the state variable
        print("Running burn-in...")
        p0 = sampler.run_mcmc(self.p0, self.nib)

        # Resets the chain to remove burn in samples and sets the walkers lower
        sampler.reset()

        # Do production. Starts from the final position of burn-in chain (rstate0 is
        # the state of the internal random number generator)
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, self.ni)

        return sampler, pos, prob, state

    def lnprior(self, theta):
        """Checks if all variables are within their priors (as well as
        determining them setting the same).
        If all priors are satisfied it needs to return '0.0' and if not '-np.inf'"""
        fwhm = theta
        if fwhm < 100. and fwhm > 0.1:
            return 0.0
        else:
            return -np.inf

    def lnlike(self, theta: np.ndarray, vis2data: np.ndarray, vis2err:
               np.ndarray, *args, **kwargs):
        """Takes theta vector and the x, y and the yerr of the theta.
        Returns a number corresponding to how good of a fit the model is to your
        data for a given set of parameters, weighted by the data points.  That it is more important"""
        if self.numerical:
            visdatamod, visdataphase, vis_axis, vis_scaling = self.model4fit_numerical(theta, *args, **kwargs)
        else:
            visdatamod = self.model4fit_analytical(theta, *args, **kwargs)

        vis2datamod = visdatamod*np.conj(visdatamod)
        return -0.5*np.sum((vis2data-vis2datamod)**2/vis2err)


    def lnprob(self, theta: np.ndarray, vis2data, vis2err, *args, **kwargs) -> np.ndarray:
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
        lp = self.lnprior(theta)
        if not lp == 0.0:
            return -np.inf
        return lp + self.lnlike(theta, vis2data, vis2err, *args, **kwargs)

    def model4fit_analytical(self, theta: np.ndarray, *args, **kwargs) -> np.ndarray:
        """The analytical model defined for the fitting process."""
        model_vis = self.model.eval_vis(theta, *args, **kwargs)
        return model_vis

    def model4fit_numerical(self, theta: np.ndarray, *args, **kwargs) -> np.ndarray:
        """The model image, that is fourier transformed for the fitting process"""
        model_img = self.model.eval_model(theta, *args, **kwargs)
        fourier = FFT(model_img, args[3])
        ft, amp, phase = fourier.pipeline()       # The wavelength should be args[3]
        return amp, phase, fourier.fftfreq, fourier.fftscale

    def get_best_fit(self, sampler):
        """Fetches the best fit values from the sampler"""
        samples = sampler.get_chain(flat=True)
        theta_max = samples[np.argmax(sampler.flatlnprobability)][0]
        return theta_max

    def plot_model_and_vis_curve(self, sampler, *args, **kwargs) -> None:
        """Plot the samples to get estimate of the density that has been sampled, to
        test if sampling went well"""
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Gets the best value and calculates a full model
        theta_max = self.get_best_fit(sampler)
        if self.numerical:
            ...
        else:
            best_fit_model = self.model.eval_vis(theta_max, *args, **kwargs)

        # Gets the model size and takes a slice of the middle for both vis2 and
        # baselines
        size_model = len(best_fit_model)
        u, v = self.model.axis_vis, self.model.axis_vis[:, np.newaxis]
        xvis_curve = np.sqrt(u**2+v**2)[size_model//2]
        yvis_curve = best_fit_model[size_model//2]
        wavelength = trunc(args[1]*1e06, 2)

        # Combines the plots and gives descriptions to the axes
        ax1.imshow(best_fit_model)
        ax1.set_title(self.model.name + fr" model at {wavelength}$\mu$m")
        ax1.set_xlabel(f"Resolution of {args[0]} px")
        ax2.errorbar(xvis_curve, yvis_curve)
        ax2.set_xlabel("Projected baselines [m]")
        ax2.set_ylabel("Vis2")
        plt.show()
        # plt.savefig("Gauss2D_to_model_data_fit.png")

    def plot_corner(self, sampler):
        """Plots the corner plot of the posterior spread"""
        labels = ["FWHM"]
        samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
        fig = corner.corner(samples, labels=labels)
        plt.show()

    def test_model(self, sampler) -> None:
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

if __name__ == "__main__":
    # Set the initial values for the parameters 
    # fwhm = 19.01185824931766         # Fitted value to model data
    fwhm = 1.
    initial = np.array([fwhm])

    # Readout Fits for real data
    f = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(f)

    # Sets the wavelength the model is to be evaluated at
    wl_ind = 101
    sampling, wavelength = 128, readout.get_wl()[wl_ind]

    # Real vis2data and error
    vis2data, vis2err = readout.get_vis24wl(wl_ind)
    # Test the fitting by inputting model data
    # vis2data, vis2err = np.load("model_fit_test.npy"), np.mean(np.load("model_fit_test.npy"))*0.02
    uvcoords = readout.get_uvcoords()

    # The number of walkers (must be even) and the number of dimensions/parameters
    nwalkers, ndim = 10, len(initial)

    # Sets the steps of the burn-in and the max. steps
    niter_burn, niter = 100, 1000

    # This vector defines the starting points of each walker for the amount of
    # dimensions
    p0 = np.random.rand(nwalkers, ndim)

    # Set the mcmc parameters and the the data to be fitted.
    mc_params = (p0, nwalkers, ndim, niter_burn, niter)
    # Set the data, the wavlength has to be the fourth argument [3]
    data = (vis2data, vis2err, sampling, wavelength, uvcoords)

    # This calls the MCMC fitting
    mcmc = MCMC(Gauss2D, mc_params, data, numerical=False)
    mcmc.pipeline()


