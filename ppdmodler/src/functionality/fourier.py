#!/usr/bin/env python3

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import numpy as np
import time
import matplotlib.pyplot as plt

from numpy import fft
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

# Own modules
from src.models import Gauss2D, Ring, InclinedDisk
from src.functionality.utilities import timeit, get_px_scaling, zoom_array

# TODO: Make class and function documentation

class FFT:
    """A collection of the fft-functionality given by scipy

    Attributes
    ----------

    """
    # TODO: Make Zoom function work
    def __init__(self, model: np.array, wavelength: float, set_size: int = None,
                 pixelscale: float = 1.) -> None:
        self.model = model
        self.model_size = len(self.model)
        self.set_size = set_size

        self.fftfreq = fft.fftfreq(self.model_size, d=pixelscale)

    def pipeline(self, zoom: bool = False) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Combines various functions and executes them

        Parameters
        ----------
        zoom: bool, optional
            Zooms the FFT to remove the zero padding
        vis: bool
            The normed FFT by its max. amplitude center output

        Returns
        -------
        ft: np.ndarray
            The FFT
        corr_flux/vis: np.ndarray
            Either returns the unnormed correlated fluxes or the normed
            visibilities depending on the 'vis' parameter setting
        phase: np.ndarray
            The phase of the FFT
        """
        if zoom:
            # TODO: This is broken as of 22.02.22 -> Fix
            ft = zoom_array(self.do_fft2())
        else:
            ft, ft_raw  = self.do_fft2()

        amp, phase = np.abs(ft), np.angle(ft, deg=True)

        # Norms the vis
        amp /= np.abs(ft_raw[0, 0])

        return ft, amp, phase

    @timeit
    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT and shifts the centre to the
        middle

        Returns
        --------
        ft: np.ndarray
        ft_raw: np.ndarray
        """
        return fft.fftshift(fft.fft2(fft.ifftshift(self.model))), \
                fft.fft2(self.model)

    @timeit
    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and shifts the centre to the middle

        Returns
        -------
        ift: np.ndarray
        """
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real


if __name__ == "__main__":
    ...

