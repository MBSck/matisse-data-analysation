import os
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from src.models import CompoundModel
from src.functionality.fourier import FFT
from src.functionality.genetic_algorithm import genetic_algorithm, decode
from src.functionality.utilities import chi_sq

def get_best_fit(sampler) -> np.ndarray:
    """Fetches the best fit values from the sampler"""
    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    return theta_max

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

def plot_model_and_vis_curve(sampler, model, bb_params, model_data) -> None:
    """Plot the samples to get estimate of the density that has been sampled, to
    test if sampling went well"""
    theta_max = get_best_fit(sampler)
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(20, 10))

    model = model(*bb_params, wavelength)
    datamod, cphasemod = model4fit_numerical(theta_max)
    datamod, cphasemod = datamod, cphasemod
    model_img = model.eval_model(theta_max, pixel_size, 4097)

    realdata = np.insert(realdata, 0, realflux)
    print_values([datamod, cphasemod], [realdata, realcphase])

    # # Takes a slice of the model and shows vis2-baselines
    # size_model = len(best_fit_model)
    # u, v = (axis := np.linspace(-150, 150, sampling)), axis[:, np.newaxis]
    # wavelength = trunc(self.wavelength*1e06, 2)
    # xvis_curve = np.sqrt(u**2+v**2)[centre := size_model//2]
    # yvis_curve = best_fit_model[centre]

    ax.imshow(model_img, vmax=model._max_sub_flux,\
              extent=[pixel_size//2, -pixel_size//2,\
                     -pixel_size//2, pixel_size//2])
    ax.set_title(fr"{model_init.name}: Temperature gradient, at {wavelength*1e6:.2f}$\mu$m")
    ax.set_xlabel(f"RA [mas]")
    ax.set_ylabel(f"DEC [mas]")

    bx.errorbar(realbaselines, realdata, realdataerr,
                color="goldenrod", fmt='o', label="Observed data")
    bx.scatter(realbaselines, datamod, label="Model fit data")
    bx.set_title("Correlated fluxes [Jy]")
    bx.set_xlabel("Baselines [m]")
    bx.legend(loc="upper right")

    cx.errorbar(t3phi_baselines, realcphase, realcphaserr,
                color="goldenrod", fmt='o', label="Observed data")
    cx.scatter(t3phi_baselines, cphasemod, label="Model fit data")
    cx.set_title(fr"Closure Phases [$^\circ$]")
    cx.set_xlabel("Longest baselines [m]")
    cx.legend(loc="upper right")

    fig.tight_layout()

    save_path = f"{model_init.name}_model_after_fit_{wavelength*1e6:.2f}.png"
    plt.savefig(save_path)
    plt.show()

def plot_corner(sampler: np.ndarray, labels: List, wavelength) -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, show_titles=True, labels=labels,
                       plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    save_path = f"{model_init.name}_corner_plot_{wavelength*1e6:.2f}.png"
    plt.savefig(save_path)

def lnprob(theta: np.ndarray) -> np.ndarray:
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

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta)

def lnlike(theta: np.ndarray):
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.
    """
    datamod, cphasemod, tot_flux = model4fit_numerical(theta)

    data_chi_sq = chi_sq(realdata, sigma2corrflux, datamod)
    cphase_chi_sq = chi_sq(realcphase, sigma2cphase, cphasemod)
    whole_chi_sq = data_chi_sq + cphase_chi_sq

    return -0.5*whole_chi_sq

def lnprior(theta):
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

def model4fit_numerical(theta: np.ndarray, args) -> np.ndarray:
    """The model image, that is fourier transformed for the fitting process"""
    model_flux = model_init.eval_model(theta, pixel_size, sampling)

    fft = FFT(model_flux, wavelength, model_init.pixel_scale,
             zero_padding_order)
    amp, phase, xycoords = fft.get_uv2fft2(uvcoords, t3phi_uvcoords,
                                           corr_flux=vis, vis2=vis2,
                                           intp=intp)
    amp = np.insert(amp, 0, np.sum(model_flux))

    return amp, phase, tot_flux

def do_mcmc(mcmc_params: List, lnprob, args) -> np.array:
    """Runs the emcee fit

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    The burn-in is first run to explore the parameter space and then the
    walkers settle into the maximum of the density. The state variable is
    then passed to the production run.

    The chain is reset before the production with the state variable being
    passed. 'rstate0' is the state of the internal random number generator

    Returns
    -------
    sampler
    pos
    prob
    state
    """
    inital, nwalker, ndim, nburn, niter = mcmc_params
    sampler = emcee.EnsembleSampler(walkers, ndim, lnprob, args=args)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(initial, nburn, progress=True)
    sampler.reset()

    print("--------------------------------------------------------------")
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    ...
