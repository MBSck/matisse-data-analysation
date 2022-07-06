#!/usr/bin/env python3

"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package

...

Initial sets the theta

>>> initial = np.array([1.5, 135, 1., 1., 100., 3., 0.01, 0.7])
>>> priors = [[1., 2.], [0, 180], [0., 2.], [0., 2.], [0., 180.], [1., 10.],
              [0., 1.], [0., 1.]]
>>> labels = ["AXIS_RATIO", "P_A", "C_AMP", "S_AMP", "MOD_ANGLE", "R_INNER",
              "TAU", "Q"]
>>> bb_params = [1500, 7900, 19, 140]

File to read data from

>>> f = "../../assets/Final_CAL.fits"
>>> out_path = "../../assets"

sws is for L-band flux; timmi2 for the N-band flux

>>> flux_file = "../../assets/HD_142666_timmi2.txt"

Set the data, the wavelength has to be the fourth argument [3]

>>> data = set_data(fits_file=f, flux_file=flux_file, pixel_size=100,
                    sampling=128, wl_ind=38, zero_padding_order=3, vis2=False)

Set the mcmc parameters and the data to be fitted.

>>> mc_params = set_mc_params(initial=initial, nwalkers=50, niter_burn=100,
                              niter=250)

This calls the MCMC fitting

>>> fitting = ModelFitting(CompoundModel, data, mc_params, priors, labels,
                           numerical=True, vis=True, modulation=True,
                           bb_params=bb_params, out_path=out_path)
>>> fitting.pipeline()

"""


import os
import emcee
import corner
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from src.models import CompoundModel
from src.functionality.fourier import FFT
from src.functionality.utilities import chi_sq, get_rndarr_from_bounds,\
        plot_txt
from src.functionality.readout import ReadoutFits, read_single_dish_txt2np
from src.functionality.genetic_algorithm import genetic_algorithm, decode

# TODO: Make function that randomly assigns starting parameters from priors

# FIXME: The code has some difficulties rescaling for higher pixel numbers
# and does in that case not approximate the right values for the corr_fluxes,
# see pixel_scaling

# TODO: Implement global parameter search algorithm (genetic algorithm)

# TODO: Implement optimizer algorithm

# TODO: Make plots of the model + fitting, that show the visibility curve, and
# two more that show the fit of the visibilities and the closure phases to the
# measured one

# TODO: Make one plot that shows a model of the start parameters and the
# uv-points plotted on top, before fitting starts and options to then change
# them in order to get the best fit (Even multiple times)

def generate_valid_guess(initial: List, priors: List,
                         nwalkers: int, frac: float) -> np.ndarray:
    """Generates a valid guess that is in the bounds of the priors for the
    start of the MCMC-fitting

    Parameters
    ----------
    inital: List
        The inital guess
    priors: List
        The priors that constrain the guess
    nwalkers: int
        The number of walkers to be initialised for

    Returns
    -------
    p0: np.ndarray
        A valid guess for the number of walkers according to the number of
        dimensions
    """
    proposal = np.array(initial)
    prior_dynamic = np.array([np.ptp(i) for i in priors])
    dyn = 1/prior_dynamic

    # NOTE: Switch to np.rand.normal as it gives negative values as well
    guess_lst = []
    for i in range(nwalkers):
        guess = proposal + frac*dyn*np.random.normal(proposal, dyn)
        guess_lst.append(guess)

    return guess_lst

def get_data(fits_file: Path, model, pixel_size: int,
             sampling: int, flux_file: Path = None,
             wl_ind: Optional[int] = None,
             zero_padding_order: Optional[int] = 2,
             bb_params: Optional[List] = [],
             priors: Optional[List] = [],
             vis2: Optional[bool] = False,
             intp: Optional[bool] = False) -> List:
    """Fetches the required info from the '.fits'-files and then returns a
    tuple containing it

    Parameters
    ----------
    fits_file: Path
        The '.fits'-file containing the data of the object
    model
        The model that is to be calculated
    pixel_size: int
        The size of the FOV, that is used
    sampling: int
        The amount of pixels used in the model image
    flux_file: Path, optional
        An additional '.fits'-file that contains the flux of the object
    wl_ind: int, optional
        If specified, picks one specific wavelength by its index
    zero_padding_order: int, optional
        The order of the zero padding
    bb_params: List
        The blackbody parameters that are used for Planck's law
    priors: List, optional
        The priors that set the bounds for the fitting algorithm
    intp: bool, optional
        Determines if it interpolates or rounds to the nearest pixel

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
            # FIXME: Check if there is real fluxerr -> If yes, add it
            flux = read_single_dish_txt2np(flux_file, wavelength)[wavelength[wl_ind]]
            fluxerr = None
        else:
            flux, fluxerr = readout.get_flux4wl(wl_ind)

        # NOTE: Fluxerr is just 20% of flux, if no fluxerr is given
        vis = np.insert(vis, 0, flux)
        viserr = np.insert(viserr, 0, fluxerr) if fluxerr is not None else \
                np.insert(viserr, 0, flux*0.2)

        wavelength = wavelength[wl_ind]
    else:
        if vis2:
            vis, viserr = readout.get_vis2()
        else:
            vis, viserr = readout.get_vis()

        cphase, cphaseerr = readout.get_t3phi()

        if flux_file:
            flux, fluxerr = read_single_dish_txt2np(flux_file, wavelength), None
        else:
            flux, fluxerr = readout.get_flux()

        # NOTE: Fluxerr is just 20% of flux, if no fluxerr is given
        for i, o in enumerate(vis):
            o  = np.insert(o, 0, flux[i])
            viserr[i] = np.insert(viserr[i], 0, fluxerr[i]) \
                    if fluxerr[i] is not None else \
                    np.insert(viserr[i], 0, flux[i]*0.2)

    uvcoords = readout.get_uvcoords()
    u, v = readout.get_split_uvcoords()
    t3phi_uvcoords = readout.get_t3phi_uvcoords()

    data = ([vis, cphase], [viserr, cphaseerr])
    uvcoords_lst = [uvcoords, u, v, t3phi_uvcoords]
    model_param_lst = [model, pixel_size, sampling,
                       wavelength, zero_padding_order,
                       bb_params, priors]
    vis_lst = [not vis2, vis2, intp]

    return [data, model_param_lst, uvcoords_lst, vis_lst]

def print_values(realdata: List, datamod: List, theta_max: List) -> None:
    """Prints the model's values"""
    print("Best fit corr. fluxes:")
    print(datamod[0])
    print("Real corr. fluxes")
    print(realdata[0])
    print("--------------------------------------------------------------")
    print("Best fit cphase")
    print(datamod[1])
    print("Real cphase")
    print(realdata[1])
    print("--------------------------------------------------------------")
    print("Theta max:")
    print(theta_max)

def plotter(sampler, realdata: List, model_param_lst: List,
            uvcoords_lst: List, vis_lst: List, mcmc_params: List, labels: List,
            debug: Optional[bool] = False,
            plot_px_size: Optional[int] = 2**12) -> None:
    """Plot the samples to get estimate of the density that has been sampled, to
    test if sampling went well"""
    initial, nwalkers, nburn, niter = mcmc_params
    mcmc_dict = {"nwalkers": nwalkers, "burn-in steps": nburn,
                 "production steps": niter}

    amp, cphase = realdata[0]
    amperr, cphaseerr = map(lambda x: x**2, realdata[1])

    model, pixel_size, sampling, wavelength,\
            zero_padding_order, bb_params, _ = model_param_lst
    uvcoords, u, v, t3phi_uvcoords = uvcoords_lst
    vis, vis2, intp = vis_lst
    baselines = np.insert(np.sqrt(u**2+v**2), 0, 0.)
    u_t3phi, v_t3phi = t3phi_uvcoords
    t3phi_baselines = np.sqrt(u_t3phi**2+v_t3phi**2)[2]

    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
    theta_max_dict = dict(zip(labels, theta_max))

    model_cp = model(*bb_params, wavelength)
    model_flux = model_cp.eval_model(theta_max, pixel_size, sampling)
    fft = FFT(model_flux, wavelength, pixel_size/sampling,
             zero_padding_order)
    amp_mod, cphase_mod, xycoords = fft.get_uv2fft2(uvcoords, t3phi_uvcoords,
                                           corr_flux=vis, vis2=vis2, intp=intp)
    amp_mod = np.insert(amp_mod, 0, np.sum(model_flux))

    print_values([amp_mod, cphase_mod], [amp, cphase], theta_max)

    if debug:
        plot_corner(sampler, model_cp, labels, wavelength)
        plot_chains(sampler, model_cp, theta_max, labels, wavelength)

    fig, axarr = plt.subplots(2, 3, figsize=(20, 10))
    ax, bx, cx = axarr[0].flatten()
    ax2, bx2, cx2 = axarr[1].flatten()

    # # Takes a slice of the model and shows vis2-baselines
    # size_model = len(best_fit_model)
    # u, v = (axis := np.linspace(-150, 150, sampling)), axis[:, np.newaxis]
    # wavelength = trunc(self.wavelength*1e06, 2)
    # xvis_curve = np.sqrt(u**2+v**2)[centre := size_model//2]
    # yvis_curve = best_fit_model[centre]

    title_dict = {"": ""}
    text_dict = {"General params": "",
                 "---------------------": "",
                 "blackbody_params": bb_params, "FOV": pixel_size,
                 "npx": sampling, "z_pad_order": zero_padding_order,
                 "wavelength": wavelength,
                 "": "",
                 "best_fit_values": "",
                 "---------------------": "",
                 **theta_max_dict,
                 "": "",
                 "mcmc_params": "",
                 "---------------------": "",
                 **mcmc_dict}

    plot_txt(ax, title_dict, text_dict)

    bx.errorbar(baselines, amp, amperr,
                color="goldenrod", fmt='o', label="Observed data", alpha=0.6)
    bx.scatter(baselines, amp_mod, marker='X', label="Model fit data")
    bx.set_title("Correlated fluxes [Jy]")
    bx.set_xlabel("Baselines [m]")
    bx.legend(loc="upper right")

    cx.errorbar(t3phi_baselines, cphase, cphaseerr,
                color="goldenrod", fmt='o', label="Observed data", alpha=0.6)
    cx.scatter(t3phi_baselines, cphase_mod, marker='X', label="Model fit data")
    cx.set_title(fr"Closure Phases [$^\circ$]")
    cx.set_xlabel("Longest baselines [m]")
    cx.set_ylim([-180, 180])
    cx.legend(loc="upper right")

    fft.plot_amp_phase([fig, ax2, bx2, cx2], corr_flux=True,
                       uvcoords_lst=xycoords)

    plt.tight_layout()

    save_path = f"{model_cp.name}_model_after_fit_{(wavelength*1e6):.2f}.png"
    plt.savefig(save_path)
    plt.show()

def plot_corner(sampler: np.ndarray, model,
                labels: List, wavelength) -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, show_titles=True, labels=labels,
                       plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    save_path = f"{model.name}_corner_plot_{(wavelength*1e6):.2f}.png"
    plt.savefig(save_path)

def plot_chains(sampler: np.ndarray, model, theta: List,
                labels: List, wavelength) -> None:
    """Plots the chains for debugging to see if and how they converge"""
    fig, axes = plt.subplots(len(theta), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    ndim = len(theta)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    save_path = f"{model.name}_chain_plot_{(wavelength*1e6):.2f}.png"
    plt.savefig(save_path)

def model4fit_numerical(theta: np.ndarray, model_param_lst,
                        uvcoords_lst, vis_lst) -> np.ndarray:
    """The model image, that is Fourier transformed for the fitting process"""
    model, pixel_size, sampling, wavelength,\
            zero_padding_order, bb_params, _ = model_param_lst
    uvcoords, u, v, t3phi_uvcoords = uvcoords_lst
    vis, vis2, intp = vis_lst

    model = model(*bb_params, wavelength)
    model_flux = model.eval_model(theta, pixel_size, sampling)
    fft = FFT(model_flux, wavelength, pixel_size/sampling,
             zero_padding_order)
    amp, phase, xycoords = fft.get_uv2fft2(uvcoords, t3phi_uvcoords,
                                           corr_flux=vis, vis2=vis2,
                                           intp=intp)
    amp = np.insert(amp, 0, np.sum(model_flux))

    return amp, phase

def lnlike(theta: np.ndarray, realdata,
           model_param_lst, uvcoords_lst,
           vis_lst):
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.
    """
    amp, cphase = realdata[0]
    sigma2amp, sigma2cphase= map(lambda x: x**2, realdata[1])

    amp_mod, cphase_mod = model4fit_numerical(theta, model_param_lst,
                                             uvcoords_lst, vis_lst)

    amp_chi_sq = chi_sq(amp, sigma2amp, amp_mod)
    cphase_chi_sq = chi_sq(cphase, sigma2cphase, cphase_mod)

    return -0.5*(amp_chi_sq + cphase_chi_sq)

def lnprior(theta, priors):
    """Checks if all variables are within their priors (as well as
    determining them setting the same).

    If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
    This function checks for an unspecified amount of flat priors. If upper
    bound is 'None' then no upper bound is given

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

    for i, o in enumerate(priors):
        if o[1] is None:
            if o[0] < theta[i]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)
        else:
            if o[0] < theta[i] < o[1]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)

    return 0.0 if all(check_conditons) else -np.inf

def lnprob(theta: np.ndarray, realdata,
           model_param_lst, uvcoords_lst,
           vis_lst) -> np.ndarray:
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
    priors = model_param_lst[-1]
    lp = lnprior(theta, priors)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, realdata, model_param_lst, uvcoords_lst, vis_lst)

def do_mcmc(mcmc_params: List, priors,
            labels, lnprob, args,
            frac: Optional[float] = 1e-4,
            cluster: Optional[bool] = False,
            debug: Optional[bool] = False) -> np.array:
    """Runs the emcee fit

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    The burn-in is first run to explore the parameter space and then the
    walkers settle into the maximum of the density. The state variable is
    then passed to the production run.

    The chain is reset before the production with the state variable being
    passed. 'rstate0' is the state of the internal random number generator

    Parameters
    ----------
    mcmc_params: List
    priors: List
    labels: List
    lnprob
    args: List
    frac: float, optional
    cluster: bool, optional
    debug: bool, optional
    """
    initial, nwalkers, nburn, niter = mcmc_params
    p0 = generate_valid_guess(initial, priors, nwalkers, frac)
    ndim = len(initial)

    if cluster:
        from schwimmbad import MPIPool

        with MPIPool as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=args, pool=pool)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nburn, progress=True)
            sampler.reset()

            print("--------------------------------------------------------------")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
            print("--------------------------------------------------------------")

    else:
        from multiprocessing import Pool, cpu_count

        with Pool() as pool:
            ncores = cpu_count()
            print(f"Executing MCMC with {ncores} cores.")
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=args, pool=pool)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nburn, progress=True)
            sampler.reset()

            print("--------------------------------------------------------------")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
            print("--------------------------------------------------------------")

    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
    plotter(sampler, *args, mcmc_params, labels, debug=debug)


if __name__ == "__main__":
    priors = [[1., 2.], [0, 180], [0., 2.], [0, 180], [1., 10.],
              [0., 1.], [0., 1.]]
    print(np.load ("theta.npy"), "Model origin")
    # initial = np.load("theta.npy")
    initial = get_rndarr_from_bounds(priors, True)
    labels = ["AXIS_RATIO", "P_A", "C_AMP", "MOD_ANGLE", "R_INNER", "TAU", "Q"]
    bb_params = [1500, 7900, 19, 140]
    mcmc_params = [initial, 50, 25, 25]

    fits_file = "../../assets/Test_model.fits"
    out_path = "../../assets"
    # flux_file = "../../assets/HD_142666_timmi2.txt"
    flux_file = None

    data = get_data(fits_file, CompoundModel, pixel_size=30,
                    sampling=128, flux_file=flux_file, wl_ind=30,
                    zero_padding_order=1, bb_params=bb_params,
                    priors=priors, vis2=False, intp=True)

    do_mcmc(mcmc_params, priors, labels, lnprob, data,
            frac=1e-4, cluster=False, debug=True)

