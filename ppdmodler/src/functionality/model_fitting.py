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


# FIXME: The code has some difficulties rescaling for higher pixel numbers
# and does in that case not approximate the right values for the corr_fluxes,
# see pixel_scaling

# TODO: Implement global parameter search algorithm (genetic algorithm)

# TODO: Implement optimizer algorithm

# TODO: Safe the model as a '.yaml'-file? Better than '.fits'?

# TODO: Think of rewriting code so that params are taken differently. Maybe
# like in a computer game with an entity manager

# TODO: Make documentation in Evernote of this file and then remove unncessary
# comments

# TODO: Make plots of the model + fitting, that show the visibility curve, and
# two more that show the fit of the visibilities and the closure phases to the
# measured one

# TODO: Make one plot that shows a model of the start parameters and the
# uv-points plotted on top, before fitting starts and options to then change
# them in order to get the best fit (Even multiple times)

# TODO: Make save paths for every class so that you can specify where files are
# being saved to

import numpy as np

from src.functionality.readout import ReadoutFits, read_single_dish_txt2np


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
            # TODO: Check if there is real fluxerr
            flux, fluxerr = read_single_dish_txt2np(flux_file, wavelength)[wavelength[wl_ind]], None
        else:
            flux, fluxerr = readout.get_flux4wl(wl_ind)

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

    uvcoords = readout.get_uvcoords()
    u, v = readout.get_split_uvcoords()
    t3phi_uvcoords = readout.get_t3phi_uvcoords()
    data = (vis, viserr, cphase, cphaseerr, flux, fluxerr)

    return (data, pixel_size, sampling, wavelength, uvcoords,
            u, v, zero_padding_order, t3phi_uvcoords, vis2)

#def __init__(self, model, data: List, mc_params: List[float],
#             priors: List[List[float]], labels: List[str],
#             numerical: bool = True, modulation: bool = False,
#             bb_params: List = None, out_path: Path = None,
#             intp: Optional[bool] = False) -> None:
#    self.priors, self.labels = priors, labels
#    self.bb_params = bb_params
#    self.model = model
#    self.intp = intp
#
#    self.modulation = modulation
#
#    self.fr_scaling = 0
#
#    self.data, self.pixel_size, self.sampling,self.wavelength,\
#            self.uvcoords, self.u, self.v,\
#            self.zero_padding_order, self.t3phi_uvcoords, self.vis2 = data
#
#    self.numerical, self.vis = numerical, not self.vis2
#
#    self.realdata, self.realdataerr,\
#            self.realcphase, self.realcphaserr,\
#            self.realflux, self.realfluxerr = self.data
#
#    self.realdata = np.insert(self.realdata, 0, self.realflux)
#    self.realdataerr = np.insert(self.realdataerr, 0, self.realfluxerr) \
#            if self.realfluxerr is not None else \
#            np.insert(self.realdataerr, 0, self.realflux*0.2)
#
#    self.sigma2corrflux = self.realdataerr**2
#    self.sigma2cphase = self.realcphaserr**2
#
#    self.initial, self.nw, self.nd, self.nib, self.ni = mc_params
#    self.model_init = self.model(*self.bb_params, self.wavelength)
#
#    print("Non-optimised start parameters:")
#    print(self.initial)
#    print("--------------------------------------------------------------")
#
#    self.p0 = [np.array(self.initial) +\
#               1e-1*np.random.randn(self.nd) for i in range(self.nw)]
#
#    self.realbaselines = np.insert(np.sqrt(self.u**2+self.v**2), 0, 0.)
#    self.u_t3phi, self.v_t3phi = self.t3phi_uvcoords
#
#    # TODO: Check if plotting should be in effect for longest baselines
#    self.t3phi_baselines = np.sqrt(self.u_t3phi**2+self.v_t3phi**2)[2]
#    self.out_path = out_path

def main() -> None:
    """"""
    ...

if __name__ == "__main__":
    initial = np.array([1.5, 135, 1., 1., 100., 3., 0.01, 0.7])
    priors = [[1., 2.], [0, 180], [0., 2.], [0., 2.], [0., 180.], [1., 10.],
              [0., 1.], [0., 1.]]
    labels = ["AXIS_RATIO", "P_A", "C_AMP", "S_AMP", "MOD_ANGLE", "R_INNER",
              "TAU", "Q"]
    bb_params = [1500, 7900, 19, 140]

    f = "../../assets/Final_CAL.fits"
    out_path = "../../assets"
    flux_file = "../../assets/HD_142666_timmi2.txt"

    data = set_data(fits_file=f, flux_file=flux_file, pixel_size=100,
                    sampling=128, wl_ind=38, zero_padding_order=3, vis2=False)
    mc_params = set_mc_params(initial=initial, nwalkers=100, niter_burn=250,
                              niter=500)
    fitting = ModelFitting(CompoundModel, data, mc_params, priors, labels,
                           numerical=True, modulation=True,
                           bb_params=bb_params, out_path=out_path,
                           intp=False)
    fitting.pipeline()

