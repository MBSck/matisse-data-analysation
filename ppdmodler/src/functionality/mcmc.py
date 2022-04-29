#!/usr/bin/env python3

"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package"""

import os
import yaml
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import optimize
from typing import Any, Dict, List, Union, Optional

from src.models import Gauss2D, Ring, InclinedDisk, CompoundModel
from src.functionality.fourier import FFT
from src.functionality.readout import ReadoutFits, read_single_dish_txt2np
from src.functionality.utilities import trunc, correspond_uv2scale, \
        azimuthal_modulation, get_px_scaling

# Interesting stuff -> Change p0, how it gets made and more

# TODO: The code has some difficulties rescaling for higher pixel numbers
# and does in that case not approximate the right values for the corr_fluxes,
# see pixel_scaling

# TODO: Safe the model as a '.yaml'-file? Better than '.fits'?

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

# Functions

def set_data(fits_file: Path, pixel_size: int,
             sampling: int, flux_file: Path = None,
             wl_ind: Optional[int] = None,
             vis2: Optional[bool] = False,
             zero_padding_order: Optional[int] = 4) -> List:
    """Fetches the required info from the '.fits'-files and then returns a
    tuple containing it

    Parameters
    ----------
    fits_file: Path
        The '.fits'-file containing the data of the object
    pixel_size: int
        The size of the FOV, that is used
    sampling: int
        The amount of pixels used in the model image
    flux_file: Path, optional
        An additional '.fits'-file that contains the flux of the object
    wl_ind: int, optional
        If specified, picks one specific wavelength by its index
    vis: bool, optional
        If specified, gets the vis2 data, if not gets the vis/corr_flux data

    Returns
    -------
    tuple
        The data required for the mcmc-fit, in the format (data, dataerr,
        pixel_size, sampling, wavelength, uvcoords, flux, u, v)
    """
    readout = ReadoutFits(fits_file)
    wavelength = readout.get_wl()

    if wl_ind:
        if vis2:
            vis, viserr = readout.get_vis24wl(wl_ind)
        else:
            vis, viserr = readout.get_vis4wl(wl_ind)

        cphase, cphaseerr = readout.get_t3phi4wl(wl_ind)


        if flux_file:
            flux = read_single_dish_txt2np(flux_file, wavelength)[wavelength[wl_ind]]
        else:
            flux = readout.get_flux4wl(wl_ind)

        wavelength = wavelength[wl_ind]
    else:
        if vis2:
            vis, viserr = readout.get_vis2()
        else:
            vis, viserr = readout.get_vis()

        cphase, cphaseerr = readout.get_t3phi()

        if flux_file:
            flux = read_single_dish_txt2np(flux_file, wavelength)
        else:
            flux = readout.get_flux()

    uvcoords = readout.get_uvcoords()
    u, v = readout.get_split_uvcoords()
    t3phi_baselines = readout.get_t3phi_uvcoords()
    data = (vis, viserr, cphase, cphaseerr)

    return (data, pixel_size, sampling, wavelength, uvcoords, flux,
            u, v, zero_padding_order, t3phi_baselines)

def set_mc_params(initial: np.ndarray, nwalkers: int,
                  niter_burn: int, niter: int) -> List:
    """Sets the mcmc parameters. The p0 vector defines the starting points of
    each walker for the amount of dimensions with an almost negligible offset

    Parameters
    ----------
    initial: List
        Contains the initial values of the parameters to be fitted
    nwalker: int
        The amount of walkers. Should always be a least twice that of the
        parameters
    niter_burn: int
        The amount of burn in steps until the production run starts
    niter: int
        The amount of production steps

    Returns
    -------
    tuple
        A tuple that contains (p0, nwalkers, ndim, niter_burn, niter)
    """
    ndim = len(initial)
    # p0 = [np.array(initial) + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
    return (initial, nwalkers, ndim, niter_burn, niter)


class MCMC:
    """"""
    def __init__(self, model, data: List, mc_params: List[float],
                 priors: List[List[float]], labels: List[str],
                 numerical: bool = True, vis: bool = False,
                 modulation: bool = False, bb_params: List = None,
                 out_path: Path = None) -> None:
        self.priors, self.labels = priors, labels
        self.bb_params = bb_params

        self.numerical, self.vis, self.vis2 = numerical, vis, not vis
        self.modulation = modulation

        self.fr_scaling = 0

        self.data, self.pixel_size, self.sampling,self.wavelength,\
                self.uvcoords, self.realflux, self.u, self.v,\
                self.zero_padding_order, self.t3phi_uvcoords = data

        self.realdata, self.realdataerr,\
                self.realcphase, self.realcphaserr = self.data

        self.realdata = np.insert(self.realdata, 0, self.realflux)
        self.realdataerr = np.insert(self.realdataerr, 0, np.mean(self.realdataerr))

        self.sigma2corrflux = self.realdataerr**2
        self.sigma2cphase = self.realcphaserr**2

        self.model = model(*self.bb_params, self.wavelength)

        self.initial, self.nw, self.nd, self.nib, self.ni = mc_params
        self.p0_full = self.optimise_inital_theta()
        self.p0 = self.p0_full.x
        print(self.p0)

        self.realbaselines = np.insert(np.sqrt(self.u**2+self.v**2), 0, 0.)
        self.u_t3phi, self.v_t3phi = self.t3phi_uvcoords
        self.t3phi_baselines = np.sqrt(self.u_t3phi**2+self.v_t3phi**2)

        self.out_path = out_path

    # @property
    # def xycoords(self):
    #     return [i for i in zip(self.xcoord, self.ycoord)]

    def pipeline(self) -> None:
        sampler, pos, prob, state = self.do_fit()

        # This plots the corner plots of the posterior spread
        self.plot_corner(sampler)

        # This plots the resulting model
        self.plot_model_and_vis_curve(sampler, 1025, self.wavelength)

        # This saves the best-fit model data and the real data
        # self.dump2yaml()

    def optimise_inital_theta(self):
        """Run a scipy optimisation on the initial values to get theta"""
        return optimize.minimize(self.lnlike, x0=self.initial)

    def do_fit(self) -> np.array:
        """The main pipline that executes the combined mcmc code and fits the
        model"""
        # The EnsambleSampler gets the parameters. The args are the args put into the
        # lob_prob function. Additional parameter a can be used for the stepsize. None is
        # the default
        sampler = emcee.EnsembleSampler(self.nw, self.nd, self.lnprob)

        # Burn-in of the sampler. Explores the parameter space. The walkers get settled
        # into the maximum of the density. Saves the walkers in the state variable
        print("Running burn-in...")
        p0 = sampler.run_mcmc(self.p0, self.nib, progress=True)
        print("--------------------------------------------------------------")

        # Resets the chain to remove burn in samples and sets the walkers lower
        sampler.reset()

        # Do production. Starts from the final position of burn-in chain (rstate0 is
        # the state of the internal random number generator)
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, self.ni, progress=True)
        print("--------------------------------------------------------------")

        return sampler, pos, prob, state

    def lnprob(self, theta: np.ndarray) -> np.ndarray:
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

        return lp + self.lnlike(theta)

    def lnlike(self, theta: np.ndarray):
        """Takes theta vector and the x, y and the yerr of the theta.
        Returns a number corresponding to how good of a fit the model is to your
        data for a given set of parameters, weighted by the data points.  That it is more important"""
        tau, q = theta[-2:]
        datamod, cphasemod = self.model4fit_numerical(tau, q, theta[:-2])
        tot_flux = self.model.get_total_flux(tau, q)
        datamod = np.insert(datamod, 0, tot_flux)

        data_chi_sq = np.sum((self.realdata-datamod)**2/self.sigma2corrflux)
        phase_chi_sq = np.sum((self.realcphase-cphasemod)**2/self.sigma2cphase)
        whole_chi_sq = data_chi_sq + phase_chi_sq

        return -0.5*whole_chi_sq

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

        return 0.0 if all(check_conditons) else -np.inf

    def model4fit_numerical(self, tau, q, theta: np.ndarray) -> np.ndarray:
        """The model image, that is fourier transformed for the fitting process"""
        model_img = self.model.eval_model(theta, self.pixel_size, self.sampling)
        model_flux = self.model.get_flux(tau, q)

        fft = FFT(model_flux, self.wavelength, self.model.pixel_scale,
                 self.zero_padding_order)
        amp, phase = fft.interpolate_uv2fft2(self.uvcoords, self.t3phi_uvcoords, True)

        # TODO: Use this as a test for the interpolation
        # rescaling of the uv-coords to the corresponding image size
        # self.fr_scaling = get_px_scaling(fr.fftfreq, wavelength,
        #                             self.model._mas_size, self.model._sampling)
        # self.xcoord, self.ycoord = correspond_uv2scale(self.fr_scaling, fr.model_size//2, uvcoords)
        # amp = np.array([amp[j, i] for i, j in zip(self.xcoord, self.ycoord)])
        return amp, phase

    def get_best_fit(self, sampler) -> np.ndarray:
        """Fetches the best fit values from the sampler"""
        samples = sampler.flatchain
        theta_max = samples[np.argmax(sampler.flatlnprobability)]
        return theta_max

    def plot_model_and_vis_curve(self, sampler, sampling, wavelength) -> None:
        """Plot the samples to get estimate of the density that has been sampled, to
        test if sampling went well"""
        self.theta_max = self.get_best_fit(sampler)
        fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(20, 10))

        tau, q = self.theta_max[-2:]
        self.sampling, self.wavelength = sampling, wavelength

        datamod, cphasemod = self.model4fit_numerical(tau, q, self.theta_max[:-2])
        self.datamod, self.cphasemod = datamod, cphasemod
        model_img = self.model.eval_model(self.theta_max[:-2],
                                               self.pixel_size, sampling)
        self.total_flux_fit = self.model.get_total_flux(tau, q)
        self.datamod = np.insert(self.datamod, 0, self.total_flux_fit)

        model_flux = self.model.get_flux(tau, q)

        # Correspond the best fit to the uv coords
        print("Best fit corr. fluxes:")
        print(datamod)
        print("Real corr. fluxes")
        print(self.realdata[1:])
        print("--------------------------------------------------------------")
        print("Best fit cphase")
        print(cphasemod)
        print("Real cphase")
        print(self.realcphase)
        print("--------------------------------------------------------------")
        print("Real flux:", self.realflux, "- Best fit flux:", self.total_flux_fit)
        print("--------------------------------------------------------------")
        print("Theta max:")
        print(self.theta_max)
        # # Takes a slice of the model and shows vis2-baselines
        # size_model = len(best_fit_model)
        # u, v = (axis := np.linspace(-150, 150, sampling)), axis[:, np.newaxis]
        # wavelength = trunc(self.wavelength*1e06, 2)
        # xvis_curve = np.sqrt(u**2+v**2)[centre := size_model//2]
        # yvis_curve = best_fit_model[centre]

        # Combines the plots and gives descriptions to the axes
        ax.imshow(model_flux, vmax=self.model._max_sub_flux,\
                  extent=[self.pixel_size//2, -self.pixel_size//2,\
                         -self.pixel_size//2, self.pixel_size//2])
        ax.set_title(fr"{self.model.name}: Temperature gradient, at {wavelength*1e6:.2f}$\mu$m")
        ax.set_xlabel(f"RA [mas]")
        ax.set_ylabel(f"DEC [mas]")

        bx.errorbar(self.realbaselines, self.realdata, self.realdataerr,
                    color="goldenrod", fmt='o', label="Real data")
        bx.scatter(self.realbaselines, self.datamod, label="Fit data")
        bx.set_title("Correlated fluxes [Jy]")
        bx.set_xlabel("Baselines [m]")
        bx.legend(loc="upper right")

        cx.errorbar(self.t3phi_baselines, self.realcphase, self.realcphaserr,
                    color="goldenrod", fmt='o', label="Real data")
        cx.scatter(self.t3phi_baselines, self.cphasemod, label="Fit data")
        cx.set_title(fr"Closure Phases [$^\circ$]")
        cx.set_xlabel("Baselines [m]")
        cx.legend(loc="upper right")
        save_path = f"{self.model.name}_model_after_fit_{wavelength*1e6:.2f}.png"

        if self.out_path is None:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.out_path, save_path))
        plt.show()

    def plot_corner(self, sampler) -> None:
        """Plots the corner plot of the posterior spread"""
        samples = sampler.get_chain(flat=True)  # Or sampler.flatchain
        fig = corner.corner(samples, labels=self.labels)
        save_path = f"{self.model.name}_corner_plot_{self.wavelength*1e6:.2f}.png"
        if self.out_path is None:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.out_path, save_path))

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

    def dump2yaml(self) -> None:
        """Dumps the resulting data of the model into a '.yaml'-file"""
        # TODO: Make this work
        if self.vis:
            vis_dict = {"visamp": self.realdata[0].tolist(),
                        "visphase": self.realdata[1].tolist(),
                        "total_flux@wl": self.realflux}
        else:
            vis_dict = {"vis2data": self.realdata.tolist()}

        fit_dict = {"datamod": self.datamod.tolist(),
                    "total_flux": self.total_flux_fit,
                    "Best fit - theta": self.theta_max,
                    "labels": self.labels,}

        model_dict = {self.model.name: {"Real data": vis_dict, "Fit values": fit_dict}}
        save_path = os.path.join(self.out_path, f"{self.model.name}_model_after_fit.yaml")
        with open(save_path, "w") as fy:
            yaml.safe_dump(model_dict, fy)


if __name__ == "__main__":
    # TODO: make the code work for the compound model make the compound model
    # work
    # Initial sets the theta
    initial = np.array([0.2, 180, 1., 1., 6.,  0.05, 0.7])
    priors = [[0., 1.], [0, 360], [0., 2.], [0., 2.], [1., 20.], [0., 1.], [0., 1.]]
    labels = ["AXIS_RATIO", "P_A", "C_AMP", "S_AMP", "R_INNER", "TAU", "Q"]
    bb_params = [1500, 7900, 19, 140]

    # File to read data from
    # f = "../../assets/Final_CAL.fits"
    f = "../../assets/HD_142666_2019-05-14T05_28_03_N_TARGET_FINALCAL_INT.fits"
    out_path = "../../assets"
    flux_file = "../../assets/HD_142666_timmi2.txt"

    # Set the data, the wavelength has to be the fourth argument [3]
    data = set_data(fits_file=f, flux_file=flux_file, pixel_size=100,
                    sampling=129, wl_ind=50, zero_padding_order=3)

    # Set the mcmc parameters and the data to be fitted.
    mc_params = set_mc_params(initial=initial, nwalkers=20, niter_burn=20,
                              niter=20)

    # This calls the MCMC fitting
    mcmc = MCMC(CompoundModel, data, mc_params, priors, labels, numerical=True,
                vis=True, modulation=True, bb_params=bb_params, out_path=out_path)
    mcmc.pipeline()

