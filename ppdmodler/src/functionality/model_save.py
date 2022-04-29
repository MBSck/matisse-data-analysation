#!/usr/bin/env python3

import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from src.models import CompoundModel
from src.functionality.fourier import FFT
from src.functionality.readout import ReadoutFits

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
    mod = model(*bb_params)
    model_image = mod.eval_model(theta, mas_size, px_size)
    flux = mod.get_flux(theta[:-2])
    # fft = FFT(flux, , mod.pixel_scale, 3)

def save_model(output_name: Path, sample_fits_file: Path):
    """Saves the model as a '.fits'-file

    Parameters
    ----------
    output_name: Path
    sample_fits_file: Path
    """
    ...


if __name__ == "__main__":
    theta = [0.4, 180, 3.5, 0.01, 0.7]

