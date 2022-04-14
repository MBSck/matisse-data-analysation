#!/usr/bin/env python3

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import time
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

# Own modules
from src.functionality.utilities import timeit, get_px_scaling, zoom_array,\
        mas2rad

# TODO: Add further documentation to the individual functions/properties
# TODO: Add all class attributes to the documentation

class FFT:
    """A collection of the fft-functionality given by scipy

    ...

    Parameters
    ----------
    model: np.ndarray
        The model image to be fourier transformed
    wavelength: float
        The wavelength at which the fourier transform should be conducted
    zero_padding: int, optional
        Sets the order of the zero padding in Default is '1'. The order is the
        power of two that is then added up with 1 to make a centre pixel
    pixelscale: float, optional
        The pixelscale for the frequency scaling of the fourier transform

    Attributes
    ----------
    model_shape
    dim
    model_centre
    zero_padding
    fftfreq
    fftscaling2m
    fftscaling2Mlambda
    zpfact: int
    """
    def __init__(self, model: np.array, wavelength: float, pixel_scale: float,
                 zero_padding_order: Optional[int] = 1) -> None:
        self.model = model
        self.model_unpadded_centre = self.model.shape[0]//2
        self.wl = wavelength
        self.pixel_scale = mas2rad(pixel_scale)
        self.zero_padding_order = zero_padding_order

        self.zpfact = 1

    @property
    def model_shape(self):
        return self.model.shape

    @property
    def dim(self):
        return self.model_shape[0]

    @property
    def model_centre(self):
        return self.dim//2

    @property
    def zero_padding(self):
        return 2**int(math.log(self.dim-1, 2)+self.zero_padding_order)+1

    @property
    def fftfreq(self):
        return np.fft.fftfreq(self.dim*self.zpfact, self.pixel_scale)

    @property
    def fftscaling2m(self):
        return get_px_scaling(np.roll(self.fftfreq, self.dim//2), self.wl)

    @property
    def fftscaling2Mlambda(self):
        return self.fftscaling2m/(self.wl*1e6)

    @property
    def fftaxis_m(self):
        return self.fftscaling2m*self.dim//2

    @property
    def fftaxis_Mlambda(self):
        return self.fftscaling2Mlambda*self.dim//2

    def zero_pad_model(self):
        """This adds zero padding to the model image before it is transformed
        to increase the sampling in the FFT image

        As model image is in format of a power of 2 + 1 to ensure a centre
        pixel this shows in the code
        """
        padded_image = np.zeros((self.zero_padding, self.zero_padding))
        self.pad_centre = padded_image.shape[0]//2
        self.mod_min, self.mod_max = self.pad_centre-self.model_centre,\
                self.pad_centre+self.model_centre+1

        padded_image[self.mod_min:self.mod_max,
                     self.mod_min:self.mod_max] = self.model
        return padded_image

    def interpolate_uv2fft2(self, uvcoords: np.ndarray):
        grid = (self.fftfreq, self.fftfreq)
        real=interpolate.interpn(grid, np.real(fft2D), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        imag=interpolate.interpn(grid, np.imag(fft2D), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        return real+imag*1j

    def pipeline(self) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Combines various functions and executes them

        Returns
        -------
        ft: np.ndarray
            The FFT of the model image
        vis: np.ndarray
            The normed visibilities
        phase: np.ndarray
            The phase
        """
        self.model = self.zero_pad_model()
        ft = zoom_array(self.do_fft2(), [self.mod_min, self.mod_max])
        # ft = self.do_fft2()

        amp, phase = np.abs(ft)/np.abs(ft[self.model_unpadded_centre,
                                          self.model_unpadded_centre]),\
                np.angle(ft, deg=True)

        return ft, amp, phase

    def ft_translate(self, uvcoord, vcoord, x=0, y=0):
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
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.model), s=s))


if __name__ == "__main__":
    ...
