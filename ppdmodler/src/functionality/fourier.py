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
from src.models import Gauss2D, Ring
from src.functionality.utilities import timeit, get_px_scaling, zoom_array

# TODO: Make class and function documentation

class FFT:
    """A collection of the fft-functionality given by scipy

    Attributes
    ----------

    """
    # TODO: Make Zoom function work
    def __init__(self, model: np.array, wavelength: float, set_size: int = None,
                 pixelscale: float = 1., greyscale: bool = False) -> None:
        self.model = model
        self.model_size = len(self.model)
        self.set_size = set_size

        self.fftfreq = fft.fftfreq(self.model_size, d=pixelscale)
        self.fftscale = get_px_scaling(self.fftfreq, wavelength)

    def pipeline(self, zoom: bool = False)-> [np.ndarray, np.ndarray, np.ndarray]:
        """Combines various functions and executes them"""
        if zoom:
            # TODO: This is broken as of 22.02.22 -> Fix
            ft = zoom_array(self.do_fft2())
        else:
            ft, ft_raw  = self.do_fft2()

        amp, phase = self.get_ft_amp_phase(ft, ft_raw)
        return ft, amp, phase

    @timeit
    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT
        Returns
        --------
        ft: np.ndarray
        ft_raw: np.ndarray
        """
        return fft.fftshift(fft.fft2(fft.ifftshift(self.model))), \
                fft.fft2(self.model)

    @timeit
    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and returns the inverse 2D-FFT"""
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real

    def get_ft_amp_phase(self, ft: np.ndarray, ft_raw: np.ndarray) -> [np.array, np.array]:
        """Splits the real and imaginary part of the 2D-FFT into amplitude-,
         and phase-spectrum"""
        amp = np.abs(ft)/np.abs(ft_raw[0, 0])   # Figure out why this is done?
        return amp, np.angle(ft, deg=True)

if __name__ == "__main__":
    ring = Ring()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    size = 2048
    ft, amp, phase = FFT(model := ring.eval_model([20., 45, 45, 0], size), 1.25e-05).pipeline()
    ax1.imshow(model)
    ax2.imshow(amp)
    plt.show()
