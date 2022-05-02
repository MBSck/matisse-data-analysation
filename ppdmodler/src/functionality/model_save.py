#!/usr/bin/env python3

import numpy as np

from pathlib import Path
from astropy.io import fits
from shutil import copyfile
from typing import Any, Dict, List, Union, Optional

from src.models import CompoundModel
from src.functionality.fourier import FFT
from src.functionality.readout import ReadoutFits
from src.functionality.utilities import progress_bar

# TODO: Make the model save the numbers of pixel and zero padding that was used
# for the calculation and some more edge data

def loop_model4wl(model, theta: List, bb_params: List,
                  mas_size: int, px_size: int, fits_file: Path) -> np.ndarray:
    """Loops a model for the input parameters and returns the slices of the
    wavelength

    Parameters
    ----------
    model
    theta: List
    bb_params: List
    mas_size: int
    px_size: int
    fits_file: Path

    Returns
    -------
    """
    readout = ReadoutFits(fits_file)
    wl = readout.get_wl()
    uvcoords, t3phi_uvcoords = readout.get_uvcoords(),\
            readout.get_t3phi_uvcoords()
    amp_lst, phase_lst, flux_lst = [[] for i in range(6)], [[] for i in range(4)], []

    print("Model is being calculated!")
    progress_bar(0, len(wl))
    for i, o in enumerate(wl):
        mod = model(*bb_params, o)
        model_image = mod.eval_model(theta[:-2], mas_size, px_size)
        flux = mod.get_flux(*theta[-2:])
        total_flux = mod.get_total_flux(*theta[-2:])
        fft = FFT(flux, o, mod.pixel_scale, 3)
        fft.pipeline()
        amp, phase = fft.interpolate_uv2fft2(uvcoords, t3phi_uvcoords, True)
        amp0, amp1, amp2, amp3, amp4, amp5 = amp
        phase0, phase1, phase2, phase3 = phase

        amp_lst[0].append(amp0)
        amp_lst[1].append(amp1)
        amp_lst[2].append(amp2)
        amp_lst[3].append(amp3)
        amp_lst[4].append(amp4)
        amp_lst[5].append(amp5)
        phase_lst[0].append(phase0)
        phase_lst[1].append(phase1)
        phase_lst[2].append(phase2)
        phase_lst[3].append(phase3)
        flux_lst.append(total_flux)
        progress_bar(i + 1, len(wl))
    print()

    flux_lst = np.array(flux_lst)
    fluxerr_lst = np.random.normal(np.mean(flux_lst)*0.3, np.std(flux_lst)*0.3, len(flux_lst))
    amperr_lst = [np.random.normal(np.mean(i)*0.2, np.std(i)*0.2, len(i)) for i in amp_lst]
    phaseerr_lst = [np.random.normal(np.mean(i), np.std(i), len(i)) for i in phase_lst]


    return (amp_lst, amperr_lst, phase_lst, phaseerr_lst, flux_lst, fluxerr_lst)

def save_model(output_path: Path, sample_fits_file: Path, data: List):
    """Saves the model as a '.fits'-file

    Parameters
    ----------
    output_path: Path
    sample_fits_file: Path
    data: List
    """
    amp, amperr, phase, phaseerr, flux, fluxerr = data

    copyfile(sample_fits_file, output_path)
    with fits.open(output_path, mode="update") as hdul:
        hdul["oi_vis"].data["visamp"] = amp
        hdul["oi_vis"].data["visamperr"] = amperr
        hdul["oi_t3"].data["t3phi"] = phase
        hdul["oi_t3"].data["t3phierr"] = phaseerr
        hdul["oi_flux"].data["fluxdata"] = flux
        hdul["oi_flux"].data["fluxerr"] = fluxerr

    print("'.fits'-file created and updated with model values")


if __name__ == "__main__":
    # Badly fitted model parameters have been used for 'real'-values
    theta = [2.86406810e-01, 1.20150360e+02, 1.19462899e+00, 8.19952919e-01,
             7.79778016e+00, 2.88112551e-02, 6.63000877e-01]
    bb_params = [1500, 7900, 19, 140]
    f = "../../assets/HD_142666_2019-05-14T05_28_03_N_TARGET_FINALCAL_INT.fits"
    data = loop_model4wl(CompoundModel, theta, bb_params, 100, 513, f)
    print(data)
    save_model("../../assets/Test_model.fits", f, data)

