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

from src.functionality.utilities import timeit, get_px_scaling, zoom_array,\
        mas2rad

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
        Sets the order of the zero padding. Default is '1'. The order is the
        power of two,  plus one (to make it odd to facilitate a centre pixel)
    pixel_scale: float, optional
        The pixel_scale for the frequency scaling of the fourier transform

    Attributes
    ----------
    model_shape
    dim
    model_centre
    fftfreq
    fftscaling2m
    fftaxis_m
    fftaxis_Mlambda
    zero_padding

    Methods
    -------
    zero_pad_model()
    interpolate_uv2fft2()
    get_amp_phase()
    do_fft2()
    pipeline()
    """
    def __init__(self, model: np.array, wavelength: float, pixel_scale: float,
                 zero_padding_order: Optional[int] = 1) -> None:
        self.model = model
        self.model_unpadded_dim = self.model.shape[0]
        self.model_unpadded_centre = self.model_unpadded_dim//2
        self.wl = wavelength
        self.pixel_scale = mas2rad(pixel_scale)
        self.zero_padding_order = zero_padding_order

    @property
    def model_shape(self):
        """Fetches the model's x, and y shape"""
        return self.model.shape

    @property
    def dim(self):
        """Fetches the model's x-dimension. Both dimensions are identical"""
        return self.model_shape[0]

    @property
    def model_centre(self):
        """Fetches the model's centre"""
        return self.dim//2

    @property
    def fftfreq(self):
        """Fetches the FFT's frequency axis, scaled according to the
        pixel_scale (determined by the sampling and the FOV) as well as the
        zero padding factor"""
        return np.fft.fftfreq(self.zero_padding, self.pixel_scale)

    @property
    def fftscaling2m(self):
        """Fetches the FFT's scaling in meters"""
        return get_px_scaling(np.roll(self.fftfreq, self.dim//2), self.wl)

    @property
    def fftscaling2Mlambda(self):
        """Fetches the FFT's scaling in mega lambda"""
        return self.fftscaling2m/(self.wl*1e6)

    @property
    def fftaxis_m(self):
        """Gets the FFT's axis's endpoints in meter"""
        return self.fftscaling2m*self.dim//2

    @property
    def fftaxis_Mlambda(self):
        """Gets the FFT's axis's endpoints in mega lambda"""
        return self.fftscaling2Mlambda*self.dim//2

    @property
    def zero_padding(self):
        """Zero pads the model image"""
        return 2**int(math.log(self.model_unpadded_dim-1, 2)\
                      +self.zero_padding_order)+1

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

    def interpolate_uv2fft2(self, ft: np.ndarray,
                            uvcoords: np.ndarray) -> np.ndarray:
        """Interpolate the uvcoordinates to the grid of the FFT

        Parameters
        ----------
        ft: np.ndarray
            The FFT
        uvcoords: np.ndarray
            The uv-coords

        Returns
        -------
        np.ndarray
            The interpolated FFT
        """
        grid = (np.fft.fftshift(self.fftfreq),
                np.fft.fftshift(self.fftfreq))
        real=interpolate.interpn(grid, np.real(ft), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        imag=interpolate.interpn(grid, np.imag(ft), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        return real+imag*1j

    def get_amp_phase(self, ft: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Gets the amplitude and the phase of the FFT

        Parameters
        ----------
        ft: np.ndarray
            The FFT of the model image

        Returns
        --------
        amp: np.ndarray
            The normed visibilities
        phase: np.ndarray
            The phase
        """
        ft = zoom_array(ft, [self.mod_min, self.mod_max])
        amp = np.abs(ft)/np.abs(ft[self.model_unpadded_centre,
                                          self.model_unpadded_centre])
        phase = np.angle(ft, deg=True)
        return amp, phase

    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT and shifts the centre to the
        middle

        Returns
        --------
        ft: np.ndarray
        ft_raw: np.ndarray
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.model)))

    def pipeline(self) -> np.ndarray:
        """Combines various functions and executes them

        Returns
        -------
        ft: np.ndarray
            The FFT of the model image
        """
        self.model = self.zero_pad_model()
        # TODO: Without zooming the picture to coordinate scaling is scuffed
        return self.do_fft2()


if __name__ == "__main__":
    ...
