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
from src.models import Gauss2D
from src.functionality.utilities import timeit, get_px_scaling

# TODO: Make class and function documentation

class FFT:
    """A collection of the fft-functionality given by scipy

    Attributes
    ----------

    """
    # TODO: Make Zoom function work
    def __init__(self, model: np.array, wavelength: float, set_size: int = None,
                 step_size_fft: float = 1., greyscale: bool = False) -> None:
        self.model = model
        self.model_size = len(self.model)
        self.set_size = set_size

        self.fftfreq = fft.fftfreq(self.model_size, d=step_size_fft)
        self.fftscale = get_px_scaling(self.fftfreq, wavelength)

    @timeit
    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT"""
        return fft.fftshift(fft.fft2(fft.ifftshift(self.model)))

    @timeit
    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and returns the inverse 2D-FFT"""
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real

    def zoom_fft2(self) -> None:
        """This zooms in on the FFT in after zero-padding"""
        ind_low_start, ind_low_end = 0, self.set_size//2
        ind_high_start, ind_high_end = self.set_size//2, self.set_size
        self.ft = np.delete(ft[:], np.arange(ind_low_start, ind_low_end, 0))
        self.ft = np.delete(ft[:], np.arange(ind_low_start, ind_low_end, 1))
        self.ft = np.delete(ft[:], np.arange(ind_high_start, ind_high_end, 0))
        self.ft = np.delete(ft[:], np.arange(ind_high_start, ind_high_end, 1))

    def get_ft_amp_phase(ft: [np.array, int]) -> [np.array, np.array]:
        """Splits the real and imaginary part of the 2D-FFT into amplitude-,
         and phase-spectrum"""
        return np.abs(ft), np.angle(ft)

if __name__ == "__main__":
    ring = Gauss2D()
    file_path = "/Users/scheuck/Documents/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    fourier = FFT(ring.eval_model(2048, 256.1), 1024, file_path, 8e-06)
