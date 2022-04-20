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
from src.functionality.readout import ReadoutFits, read_single_dish_txt2np
from src.functionality.utilities import trunc, correspond_uv2scale, \
        azimuthal_modulation, get_px_scaling

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

class MCMC:
    """"""
    def __init__(self, model, data: List, mc_params: List[float],
                 priors: List[List[float]], labels: List[str],
                 numerical: bool = True, vis: bool = False,
                 modulation: bool = False, bb_params: List = None,
                 out_path: Path = None) -> None:
        self.data = data[:-4]
        self.priors, self.labels = priors, labels
        self.bb_params = bb_params

        self.p0, self.nw, self.nd, self.nib, self.ni = mc_params
        self.numerical, self.vis = numerical, vis
        self.vis2 = not self.vis
        self.modulation = modulation
        self.fr_scaling = 0

        self.realdata, self.realerr, self.pixel_size,\
                self.sampling, self.wavelength, self.uvcoords,\
                self.realflux, self.u, self.v, self.zero_padding_order = data

        self.model = model(*self.bb_params, self.wavelength)

        if self.vis:
            self.realdata, self.datamod = self.realdata[0], None

        self.theta_max = None
        self.x, self.y = 0, 0

        self.out_path = out_path

    @property
    def xycoords(self):
        return [i for i in zip(self.xcoord, self.ycoord)]

    def pipeline(self) -> None:
        sampler, pos, prob, state = self.do_fit()

        # This plots the corner plots of the posterior spread
        self.plot_corner(sampler)

        # This plots the resulting model
        self.plot_model_and_vis_curve(sampler, 2049, self.wavelength)

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
            datamod, phase = self.model4fit_numerical(theta[:-2], sampling, wavelength, uvcoords)
        else:
            datamod = self.model4fit_analytical(theta[:-2], sampling, wavelength, uvcoords)

        if self.vis:
            realdata, realphase = realdata
            realdataerr, realphaseerr = realerr
            tot_flux = self.model.get_total_flux(tau, q)
            datamod *= tot_flux
        else:
            datamod = datamod*np.conj(datamod)

        sigma2corrflux = realdataerr**2 + realdata**2
        sigma2totflux = self.realflux**2*0.2
        print(datamod, "datamod", '\n', realdata, "realdata")
        print(self.realflux, "real flux", tot_flux, "calculated flux")

        data_chi_sq = np.sum((realdata-datamod)**2/sigma2corrflux)
        flux_chi_sq = np.sum((self.realflux-tot_flux)**2/sigma2totflux)
        whole_chi_sq = data_chi_sq + flux_chi_sq
        print(data_chi_sq/whole_chi_sq, flux_chi_sq/whole_chi_sq)

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
        model_img = self.model.eval_model(theta, self.pixel_size, sampling)

        fft = FFT(model_img, wavelength, self.model.pixel_scale,
                 self.zero_padding_order)
        ft = fft.pipeline()
        ft = fft.interpolate_uv2fft2(ft, uvcoords)
        amp, phase = fft.get_amp_phase(ft)

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
        fig, ax = plt.subplots(1, 1)
        print(self.theta_max, "Theta max")

        tau, q = self.theta_max[-2:]

        datamod, phase = self.model4fit_numerical(self.theta_max[:-2],
                                                      sampling, wavelength,
                                                      self.uvcoords)
        model_img = self.model.eval_model(self.theta_max[:-2],
                                               self.pixel_size, sampling)
        if self.vis:
            flux  = self.model.get_total_flux(tau, q)
            datamod *= flux
        else:
            datamod = datamod*np.conj(datamod)

        self.datamod = datamod

        # Correspond the best fit to the uv coords
        print("best fit data", datamod, "real data", self.realdata)

        # # Takes a slice of the model and shows vis2-baselines
        # size_model = len(best_fit_model)
        # u, v = (axis := np.linspace(-150, 150, sampling)), axis[:, np.newaxis]
        # wavelength = trunc(self.wavelength*1e06, 2)
        # xvis_curve = np.sqrt(u**2+v**2)[centre := size_model//2]
        # yvis_curve = best_fit_model[centre]

        # Combines the plots and gives descriptions to the axes
        ax.imshow(self.model.get_flux(tau, q), vmax=self.model._max_sub_flux,\
                  extent=[self.pixel_size//2, -self.pixel_size//2,\
                         -self.pixel_size//2, self.pixel_size//2])
        ax.set_title(fr"{self.model.name}: Temperature gradient, at {wavelength*1e6}$\mu$m")
        ax.set_xlabel(f"[mas]")
        ax.set_ylabel(f"[mas]")
        # bx.plot(xvis_curve, yvis_curve)
        # bx.set_xlabel(r"$B_p$ [m]")

        # if self.vis:
        #     bx.set_ylabel("vis/corr_flux [Jy]")
        # else:
        #     bx.set_ylabel("vis2")

        # third_max = len(best_fit_model)//2 - 3

        # cx.imshow(best_fit_model, vmax=best_fit_model[third_max, third_max],\
        #           extent=[self.pixel_size//2, -self.pixel_size//2,\
        #                   -self.pixel_size//2, self.pixel_size//2])


        # cx.scatter(self.u/self.fr_scaling, self.v/self.fr_scaling)
        # cx.set_title(fr"{self.model.name}: Correlated fluxes,  at {wavelength}$\mu$m")
        # cx.set_xlabel(f"[mas]")
        # cx.set_ylabel(f"[mas]")

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

    # TODO: Check how the vis/vis2 data is fetched -> Is it done correctly?

    if wl_ind:
        if vis2:
            data, dataerr = readout.get_vis24wl(wl_ind)
        else:
            temp_data = readout.get_vis4wl(wl_ind)
            data, dataerr = [temp_data[0], temp_data[2]], [temp_data[1], temp_data[3]]

        if flux_file:
            flux = read_single_dish_txt2np(flux_file, wavelength)[wavelength[wl_ind]]
        else:
            flux = readout.get_flux4wl(wl_ind)

        wavelength = wavelength[wl_ind]
    else:
        if vis2:
            data, dataerr = readout.get_vis2()
        else:
            temp_data = readout.get_vis()
            data, dataerr = [temp_data[0], temp_data[2]], [temp_data[1], temp_data[3]]

        if flux_file:
            flux = read_single_dish_txt2np(flux_file, wavelength)
        else:
            flux = readout.get_flux()

    uvcoords = readout.get_uvcoords()
    u, v = readout.get_split_uvcoords()

    return (data, dataerr, pixel_size, sampling,
            wavelength, uvcoords, flux, u, v, zero_padding_order)

def set_mc_params(initial, nwalkers, niter_burn):
    """Sets the mcmc parameters. The p0 vector defines the starting points of
    each walker for the amount of dimensions with an almost negligible offset

    Parameters
    ----------
    inital: List
        Contains the initial values of the parameters to be fitted
    nwalker: int
        The amount of walkers. Should always be a least twice that of the
        parameters
    niter_burn: int
        The amount of burn in steps until the production run starts

    Returns
    -------
    tuple
        A tuple that contains (p0, nwalkers, ndim, niter_burn, niter)
    """
    ndim, niter  = len(initial), niter_burn*10
    p0 = [np.array(initial) + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]
    return (p0, nwalkers, ndim, niter_burn, niter)

if __name__ == "__main__":
    # TODO: make the code work for the compound model make the compound model
    # work
    # Initial sets the theta
    initial = np.array([0.4, 45, 0.1, 0.5])
    priors = [[0., 1.], [0, 360], [0.01, 1.], [0.01, 1.]]
    labels = ["AXIS_RATIO", "P_A", "TAU", "Q"]
    bb_params = [1500, 7900, 19, 140]

    # File to read data from
    f = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    out_path = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets"
    flux_file = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/HD_142666_timmi2.txt"

    # Set the data, the wavlength has to be the fourth argument [3]
    data = set_data(fits_file=f, flux_file=flux_file, pixel_size=30,
                    sampling=129, wl_ind=30, zero_padding_order=3)

    # Set the mcmc parameters and the the data to be fitted.
    mc_params = set_mc_params(initial=initial, nwalkers=10, niter_burn=1000)

    # This calls the MCMC fitting
    mcmc = MCMC(CompoundModel, data, mc_params, priors, labels, numerical=True,
                vis=True, modulation=True, bb_params=bb_params, out_path=out_path)
    mcmc.pipeline()

