#!/usr/bin/env python3

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import time
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interpn
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

from src.functionality.utilities import timeit, zoom_array, mas2rad

# FIXME: The scaling of the meters seems to be off for uniform disk
# and ring as well

class FFT:
    """A collection and build up on the of the FFT-functionality given by numpy

    ...

    Parameters
    ----------
    model: np.ndarray
        The model image to be fourier transformed
    wavelength: float
        The wavelength at which the fourier transform should be conducted
    pixel_scale: float
        The pixel_scale for the frequency scaling of the fourier transform
    zero_padding: int, optional
        Sets the order of the zero padding. Default is '1'. The order is the
        power of two,  plus one (to make it odd to facilitate a centre pixel)

    Attributes
    ----------
    model_shape
    dim
    model_centre
    fftfreq
    fftscaling2m
    fftaxis_m_end
    fftaxis_m
    fftaxis_Mlambda_end
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
        self.model = self.unpadded_model = model
        self.model_unpadded_dim = self.model.shape[0]
        self.model_unpadded_centre = self.model_unpadded_dim//2

        self.wl = wavelength
        self.pixel_scale = mas2rad(pixel_scale)
        self.zero_padding_order = zero_padding_order

        self.ft = self.pipeline()

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
        return np.diff(np.fft.fftshift(self.fftfreq))[0]*self.wl

    @property
    def fftscaling2Mlambda(self):
        """Fetches the FFT's scaling in mega lambda"""
        return self.fftscaling2m/(self.wl*1e6)

    @property
    def fftaxis_m_end(self):
        """Fetches the endpoint of the FFT's axis in meters"""
        return self.fftscaling2m*self.model_centre

    @property
    def fftaxis_Mlambda_end(self):
        """Fetches the endpoint of the FFT's axis in mega lambdas"""
        return self.fftscaling2Mlambda*self.model_centre

    @property
    def fftaxis_m(self):
        """Gets the FFT's axis's in meters"""
        return np.linspace(-self.fftaxis_m_end, self.fftaxis_m_end, self.dim)

    @property
    def fftaxis_Mlambda(self):
        """Gets the FFT's axis's endpoints in mega lambdas"""
        return np.linspace(-self.fftaxis_Mlambda_end, self.fftaxis_Mlambda_end, self.dim)

    @property
    def zero_padding(self):
        """The new pixel size to be had after zero padding"""
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

    def interpolate_uv2fft2(self, uvcoords: np.ndarray,
                            uvcoords_cphase: np.ndarray,
                            corr_flux: bool = False) -> np.ndarray:
        """Interpolate the uvcoordinates to the grid of the FFT

        Parameters
        ----------
        uvcoords: np.ndarray
            The uv-coords for the correlated fluxes
        uvcoords_cphase: np.ndarray
            The uv-coords for the closure phases
        corr_flux: bool
            If the input image is a temperature gradient model then set this to
            'True' and the output will be the correlated fluxes

        Returns
        -------
        amp: np.ndarray
            The interpolated amplitudes
        cphases: np.ndarray
            The interpolated closure phases
        """
        grid = (self.fftaxis_m, self.fftaxis_m)
        real_corr = interpn(grid, np.real(self.ft), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        imag_corr = interpn(grid, np.imag(self.ft), uvcoords, method='linear',
                                 bounds_error=False, fill_value=None)
        ft_intp_corr = real_corr+1j*imag_corr

        ucphase, vcphase = uvcoords_cphase
        real_cphase = interpn(grid, np.real(self.ft), uvcoords_cphase, method='linear',
                                 bounds_error=False, fill_value=None, period=180)
        imag_cphase = interpn(grid, np.imag(self.ft), uvcoords_cphase, method='linear',
                                 bounds_error=False, fill_value=None, period=180)
        cphases = sum(np.angle(real_cphase+1j*imag_cphase, deg=True))

        if corr_flux:
            return np.abs(ft_intp_corr), cphases
        return np.abs(ft_intp_corr)/np.abs(self.ft_center), cphases

    def get_amp_phase(self, corr_flux: Optional[bool] = False) -> [np.ndarray, np.ndarray]:
        """Gets the amplitude and the phase of the FFT

        Parameters
        ----------
        corr_flux: bool, optional
            If the input image is a temperature gradient model then set this to
            'True' and the output will be the correlated fluxes

        Returns
        --------
        amp: np.ndarray
            The normed visibilities
        phase: np.ndarray
            The phase
        """
        if corr_flux:
            amp, phase = np.abs(self.ft), np.angle(self.ft, deg=True)
        else:
            amp, phase  = np.abs(self.ft)/np.abs(self.ft_center),\
                    np.angle(self.ft, deg=True)

        return amp, phase

    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT and shifts the centre to the
        middle

        Returns
        --------
        ft: np.ndarray
        """
        raw_ft = np.fft.fft2(np.fft.ifftshift(self.model))
        self.ft_center = raw_ft[0][0]
        return np.fft.fftshift(raw_ft)

    def pipeline(self) -> np.ndarray:
        """Combines various functions and executes them

        Returns
        -------
        ft: np.ndarray
            The FFT of the model image
        """
        self.model = self.zero_pad_model()
        return self.do_fft2()

    def plot_amp_phase(self, matplot_axis: Optional[List] = [],
                       zoom: Optional[int] = 500,
                       corr_flux: Optional[bool] = True,
                       interpolate: Optional[bool] = False,
                       plt_save: Optional[bool] = False) -> None:
        """This plots the input model for the FFT as well as the resulting
        amplitudes and phases for units of both [m] and [Mlambda]

        Parameters
        ----------
        matplot_axis: List, optional
            The axis of matplotlib
        zoom: bool, optional
            The zoom for the (u,v)-coordinates in [m], the [Mlambda] component
            will be automatically calculated to fit
        corr_flux: bool, optional
            If the amplitudes will be normed or not
        interpolate: bool, optional
            If toggled then the interpolation will be overplotted
        plt_save: bool, optional
            Saves the plot if toggled on
        """
        if matplot_axis:
            fig, ax, bx, cx = matplot_axis
        else:
            fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
            ax, bx, cx = axarr.flatten()

        fov_scale = int((self.pixel_scale/mas2rad())*self.model_unpadded_dim)
        zoom_Mlambda = zoom/(self.wl*1e6)

        if corr_flux:
            vmax = np.sort(self.unpadded_model.flatten())[::-1][1]
        else:
            vmax = None

        amp, phase = self.get_amp_phase(corr_flux)
        ax.imshow(self.unpadded_model, vmax=vmax,
                  extent=[-fov_scale, fov_scale, -fov_scale, fov_scale])
        cbx = bx.imshow(amp, extent=[-self.fftaxis_m_end,
                               self.fftaxis_m_end, self.fftaxis_Mlambda_end,
                                 -self.fftaxis_Mlambda_end],
                  aspect=self.wl*1e6)
        ccx = cx.imshow(phase, extent=[-self.fftaxis_m_end,
                                 self.fftaxis_m_end, self.fftaxis_Mlambda_end,
                                 -self.fftaxis_Mlambda_end],
                  aspect=self.wl*1e6)


        label_vis = "Flux [Jy]" if corr_flux else "vis"
        fig.colorbar(cbx, fraction=0.046, pad=0.04, ax=bx, label=label_vis)
        fig.colorbar(ccx, fraction=0.046, pad=0.04, ax=cx, label="Phase [°]")

        ax.set_title(f"Model image at {self.wl}, Object plane")
        bx.set_title("Amplitude of FFT")
        cx.set_title("Closure phase of FFT")

        ax.set_xlabel("RA [mas]")
        ax.set_ylabel("DEC [mas]")

        bx.set_xlabel("u [m]")
        bx.set_ylabel(r"v [M$\lambda$]")

        cx.set_xlabel("u [m]")
        cx.set_ylabel(r"v [M$\lambda$]")

        bx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])
        cx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])

        fig.tight_layout()

        if plt_save:
            plt.savefig(f"{self.wl}_FFT_plot.png")
        else:
            plt.show()


if __name__ == "__main__":
    ...
