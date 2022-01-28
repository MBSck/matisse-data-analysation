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
from src.functionality.utilities import timeit, get_scaling_px2metr

# TODO: Make class and function documentation

class FFT:
    """A collection of the fft-functionality given by scipy

    Attributes
    ----------

    """
    # TODO: Make Zoom function work
    def __init__(self, model: np.array, set_size: int, fits_file_path: Path,
                 wavelength: float, step_size_fft: float = 1., greyscale: bool = False) -> None:
        self.model = model                                                           # Evaluates the model
        self.model_size = len(self.model)                                           # Gets the size of the model's image

        # General variables
        self.set_size = set_size
        self.fftfreq = fft.fftfreq(self.model_size, d=step_size_fft)                # x-axis of the FFT corresponding to px/img_size
        self.roll = np.floor(self.model_size/2).astype(int)                         # Gets half the pictures size as int
        self.freq = np.roll(self.fftfreq, self.roll, axis=0)                        # Rolls 0th-freq to centre
        self.fftscale = np.diff(self.freq)[0]                                       # cycles/mas per px in FFT img

    def fft_pipeline(self) -> [float, np.array, float, float]:
        """A pipeline function that calls the functions of the FFT in order, and
        avoids double calling of single functions

        Returns
        -------
        ft: np.array
            The fourier transform of the model
        scaling: float
            The scaling factor of the fourier transform
        """
        self.ft = self.do_fft2()
        self.scaling = get_scaling_px2metr(self.fftscale)
        # self.zoom_fft2()
        return self.ft, self.scaling

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
    fourier = FFT(ring.eval_model(1024, 256.1), 1024, file_path, 8e-06)
