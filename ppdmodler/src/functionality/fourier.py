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
from src.functionality.utilities import timeit, get_px_scaling, zoom_array

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

    def pipeline(self, zoom: bool = False)-> [np.ndarray, np.ndarray, np.ndarray]:
        """Combines various functions and executes them"""
        if zoom:
            ft = zoom_array(self.do_fft2())
        else:
            ft = self.do_fft2()

        amp, phase = self.get_ft_amp_phase(ft)
        return ft, amp, phase

    @timeit
    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT"""
        return fft.fftshift(fft.fft2(fft.ifftshift(self.model)))

    @timeit
    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and returns the inverse 2D-FFT"""
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real

    def get_ft_amp_phase(self, ft: np.array) -> [np.array, np.array]:
        """Splits the real and imaginary part of the 2D-FFT into amplitude-,
         and phase-spectrum"""
        return np.abs(ft), np.angle(ft)

if __name__ == "__main__":
    gauss = Gauss2D()
    file_path = "/Users/scheuck/Documents/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    fourier = FFT(gauss.eval_model([1., 256.1], 2048), 1024, file_path, 8e-06)
