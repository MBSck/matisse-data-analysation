#!/usr/bin/env python3

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import numpy as np
import time
import matplotlib.pyplot as plt

from numpy import fft
from scipy import interpolate
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

# Own modules
from src.functionality.utilities import timeit, get_px_scaling, zoom_array,\
        mas2rad

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
        s = model.shape
        self.dim = s[0]
        self.zpfact = 1
        self.set_size = set_size
        self.px_size = 0.1                  # [mas]

    @property
    def fftfreq(self):
        return np.roll(fft.fftfreq(self.dim*self.zpfact,
                                   mas2rad(self.px_size)), self.dim//2)

    @property
    def fftscaling_mlambda(self):
        return np.roll(fft.fftfreq(self.dim*self.zpfact,
                                   mas2rad(self.px_size)), self.dim//2)*1e6

    def interpolate_uv2fft2(self, uvcoords: np.ndarray):
        grid = (self.fftfreq, self.fftfreq)
        real=interpolate.interpn(grid, np.real(fft2D), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        imag=interpolate.interpn(grid, np.imag(fft2D), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        return real+imag*1j

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

    def ft_translate(self, uvcoord, vcoord, wavelength, x=0, y=0):
        """The translation factor for the fft"""
        return np.exp(-2j*np.pi*(uvcoord*x+vcoord*y))

    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT and shifts the centre to the
        middle

        Returns
        --------
        ft: np.ndarray
        ft_raw: np.ndarray
        """
        s = [self.zpfact*self.dim, self.zpfact*self.dim]
        return fft.fftshift(fft.fft2(fft.fftshift(self.model), s=s)), \
                fft.fft2(self.model, s=s)

    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and shifts the centre to the middle

        Returns
        -------
        ift: np.ndarray
        """
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real


if __name__ == "__main__":
    ...

