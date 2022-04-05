#!/usr/bin/env python3

"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package"""

import os
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from pathlib import Path

from typing import Any, Dict, List, Union, Optional

from src.models import Gauss2D, Ring, InclinedDisk, CompoundModel
from src.functionality.fourier import FFT
from src.functionality.readout import ReadoutFits
from src.functionality.utilities import trunc, correspond_uv2scale, \
        azimuthal_modulation

# TODO: Think of rewriting code so that params are taken differently
# TODO: Make documentation in Evernote of this file and then remove unncessary
# comments
# TODO: Make this more generally applicable
# TODO: Make plots of the model + fitting, that show the visibility curve, and
# two more that show the # fit of the visibilities and the closure phases to the measured one
# TODO: Make one plot that shows a model of the start parameters and the
# uv-points plotted on top, before fitting starts and options to then change
# them in order to get the best fit (Even multiple times)
# TODO: Make save paths for every class so that you can specify where files are
# being saved to

class MCMC:
    def __init__(self, model, data: List, mc_params: List[float],
                 priors: List[List[float]], labels: List[str],
                 numerical: bool = True, vis: bool = False,
                 modulation: bool = False, bb_params: List = None,
                 out_path: Path = None) -> None:
        self.model = model()
        self.data = data
        self.priors, self.labels = priors, labels
        self.bb_params = bb_params

        self.p0, self.nw, self.nd, self.nib, self.ni = mc_params
        self.numerical, self.vis = numerical, vis
        self.vis2 = not self.vis
        self.modulation = modulation

        self.realdata, self.realerr, self.pixel_size,\
                self.sampling, self.wavelength, self.uvcoords = data

        if self.vis:
            self.realdata, self.datamod = self.realdata[0], None

        self.theta_max = None
        self.x, self.y = 0, 0

        self.out_path = out_path

    def pipeline(self) -> None:
        sampler, pos, prob, state = self.do_fit()

        # This plots the corner plots of the posterior spread
        self.plot_corner(sampler)

        # This plots the resulting model
        self.plot_model_and_vis_curve(sampler, 2048, self.wavelength, self.uvcoords)

        # This saves the best-fit model data and the real data
        self.save_fit_data()

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

    def lnprob(self, theta: np.ndarray, realdata, realerr, pixel_size, sampling,
               wavelength, uvcoords) -> np.ndarray:
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

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnlike(theta, realdata, realerr, sampling, wavelength, uvcoords)

    def lnlike(self, theta: np.ndarray, realdata: np.ndarray, realerr:
               np.ndarray, sampling, wavelength, uvcoords):
        """Takes theta vector and the x, y and the yerr of the theta.
        Returns a number corresponding to how good of a fit the model is to your
        data for a given set of parameters, weighted by the data points.  That it is more important"""
        tau, q = theta[-2:]
        if self.numerical:
            datamod, phase, ft = self.model4fit_numerical(theta[:-2], sampling, wavelength, uvcoords)
        else:
            datamod = self.model4fit_analytical(theta[:-2], sampling, wavelength, uvcoords)

        if self.vis:
            realdata, realphase = realdata
            datamod *= self.model.get_flux(tau, q, *self.bb_params, wavelength)
        else:
            datamod = datamod*np.conj(datamod)

        print(datamod, "mod data", '\n', realdata, "real data")

        return -0.5*np.sum(((realdata-datamod)/(realdata*0.1))**2)

    def lnprior(self, theta):
        """Checks if all variables are within their priors (as well as
        determining them setting the same).

        If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
        This function checks for an unspecified amount of flat priors

        Parameters
        ----------
        theta: List
            A list of all the parameters that ought to be fitted

        Returns
        -------
        float
            Return-code 0.0 for within bounds and -np.inf for out of bound
            priors
        """
        check_conditons = []
        for i, o in enumerate(self.priors):
            if o[0] < theta[i] < o[1]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)

        if all(check_conditons):
            return 0.0
        else:
            return -np.inf

    def model4fit_analytical(self, theta: np.ndarray, sampling, wavelength,
                             uvcoords) -> np.ndarray:
        """The analytical model defined for the fitting process."""
        model_vis = self.model.eval_vis(theta, sampling, wavelength, uvcoords)
        return model_vis

    def model4fit_numerical(self, theta: np.ndarray, sampling, wavelength,
                            uvcoords) -> np.ndarray:
        """The model image, that is fourier transformed for the fitting process"""
        if self.vis:
            model_img = self.model.eval_model(theta, self.pixel_size, sampling)
        else:
            model_img = self.model.eval_model(theta, self.pixel_size, sampling)

        if self.modulation:
            model_img *= azimuthal_modulation(self.model._phi, [[1, 1]])

        fr = FFT(model_img, wavelength)
        ft, amp, phase = fr.pipeline()

        # rescaling of the uv-coords to the corresponding image size
        self.xcoord, self.ycoord = correspond_uv2scale(fr.fftscale, fr.model_size//2, uvcoords)
        amp = np.array([amp[j, i] for i, j in zip(self.xcoord, self.ycoord)])
        return amp, phase, ft

    def get_best_fit(self, sampler) -> np.ndarray:
        """Fetches the best fit values from the sampler"""
        samples = sampler.flatchain
        theta_max = samples[np.argmax(sampler.flatlnprobability)]
        return theta_max

    def plot_model_and_vis_curve(self, sampler, sampling, wavelength, uvcoords) -> None:
        """Plot the samples to get estimate of the density that has been sampled, to
        test if sampling went well"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Gets the best value and calculates a full model
        self.theta_max = self.get_best_fit(sampler)
        print(self.theta_max, "Theta max")
        tau, q = self.theta_max[-2:]

        if self.numerical:
            # For debugging only
            datamod, phase, ft = self.model4fit_numerical(self.theta_max[:-2], sampling, wavelength, uvcoords)
            if self.vis:
                flux = self.model.get_flux(tau, q, *self.bb_params, wavelength)
                datamod *= flux
            else:
                datamod = datamod*np.conj(datamod)

            self.datamod = datamod
            best_fit_model = self.model.eval_model(self.theta_max[:-2],
                                                   self.pixel_size, sampling)
            flux = self.model.get_flux(tau, q, *self.bb_params, wavelength)
            best_fit_model = abs(FFT(best_fit_model, wavelength).pipeline()[1])
            best_fit_model *= flux
        else:
            best_fit_model = self.model.eval_vis(self.theta_max, sampling, wavelength)

        # Correspond the best fit to the uv coords
        print("best fit data", datamod, "real data", self.realdata)

        # Takes a slice of the model and shows vis2-baselines
        size_model = len(best_fit_model)
        u, v = (axis := np.linspace(-150, 150, sampling)), axis[:, np.newaxis]
        wavelength = trunc(wavelength*1e06, 2)
        xvis_curve = np.sqrt(u**2+v**2)[centre := size_model//2]
        yvis_curve = best_fit_model[centre]

        # Combines the plots and gives descriptions to the axes
        ax1.imshow(best_fit_model)
        ax1.set_title(fr"{self.model.name}-model at {wavelength}$\mu$m")
        ax1.set_xlabel(f"Resolution of {sampling} px")
        ax2.errorbar(xvis_curve, yvis_curve)
        ax2.set_xlabel(r"$B_p$ [m]")

        if self.vis:
            ax2.set_ylabel("vis/corr_flux")
        else:
            ax2.set_ylabel("vis2")

        if self.out_path is None:
            plt.savefig(f"{self.model.name}_model_after_fit.png")
        else:
            plt.savefig(os.path.join(self.out_path, f"{self.model.name}_model_after_fit.png"))
        plt.show()

    def plot_corner(self, sampler) -> None:
        """Plots the corner plot of the posterior spread"""
        samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
        fig = corner.corner(samples, labels=self.labels)
        if self.out_path is None:
            plt.savefig(f"{self.model.name}_corner_plot.png")
        else:
            plt.savefig(os.path.join(self.out_path, f"{self.model.name}_corner_plot.png"))

    def test_model(self, sampler) -> None:
        # TODO: Implement printout of theta_max
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

    def write_data(self) -> str:
        """Specifies how the fit data should be written"""
        if self.vis:
            write_str = f"visamp\n{str(self.realdata[0])}\n-------------------\n"\
                    f"visphase\n{str(self.realdata[1])}\n-------------------\n"
        else:
            write_str = f"vis2data\n{str(self.realdata)}\n-------------------\n"

        write_str += f"datamod\n{str(self.datamod)}\n-------------------\n"\
                f"theta - best fit\n{str(self.theta_max)}\n-------------------\n"\
                f"labels\n{str(self.labels)}\n"
        return write_str

    def save_fit_data(self) -> None:
        """This saves the real data and the best-fitted model data"""
        if self.out_path is not None:
            with open(os.path.join(self.out_path, f"{self.model.name}_data.txt"), "w+") as f:
                f.write(self.write_data())
        else:
            with open(f"{self.model.name}_data.txt", "w+") as f:
                f.write(self.write_data())

def set_data(fits_file: Path, pixel_size: int, sampling: int, wl_ind: int, vis: bool = False):
    """Fetches the required info and then gets the uvcoords and makes the
    data"""
    readout = ReadoutFits(fits_file)

    if vis:
        temp_data = readout.get_vis4wl(wl_ind)
        data, dataerr = [temp_data[0], temp_data[2]], [temp_data[1], temp_data[3]]
    else:
        data, dataerr = readout.get_vis24wl(wl_ind)

    uvcoords = readout.get_uvcoords()
    wavelength = readout.get_wl()[wl_ind]

    return (data, dataerr, pixel_size, sampling, wavelength, uvcoords)

def set_mc_params(initial, nwalkers, ndim, niter_burn, niter):
    """Sets the mcmc parameters"""
    # This vector defines the starting points of each walker for the amount of
    # dimensions
    p0 = [np.array(initial) + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]

    return (p0, nwalkers, ndim, niter_burn, niter)

if __name__ == "__main__":
    # TODO: make the code work for the compound model make the compound model
    # work
    # Initial sets the theta
    initial = np.array([1.5, 45, 45, 45, 0.5, 0.5])
    priors = [[0., 10.], [0, 360], [0, 360], [0, 360], [0., 1.], [0., 1.]]
    labels = ["FWHM", "E_A", "P_A", "INC_A","TAU", "Q"]
    bb_params = [1500, 19, 140]

    # File to read data from
    f = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    out_path = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets"

    # Set the data, the wavlength has to be the fourth argument [3]
    data = set_data(fits_file=f, pixel_size=10, sampling=128, wl_ind=101, vis=True)

    # Set the mcmc parameters and the the data to be fitted.
    mc_params = set_mc_params(initial=initial, nwalkers=20, ndim=len(initial), niter_burn=100, niter=1000)

    # This calls the MCMC fitting
    mcmc = MCMC(CompoundModel, data, mc_params, priors, labels, numerical=True,
                vis=True, modulation=True, bb_params=bb_params, out_path=out_path)
    mcmc.pipeline()

