#!/usr/bin/env python3

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import time
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.interpolate import interpn
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

from src.functionality.utilities import timeit, zoom_array, mas2rad

# FIXME: There seems to be some problem with the phases of the FFT but not the
# amps

# TODO: Think of how to implement the odd centre point after fixing the phases

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
        self.pixel_scale = pixel_scale
        self.zero_padding_order = zero_padding_order

        self.ft = self.do_fft2()

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
    def freq_axis(self):
        """Fetches the FFT's frequency axis, scaled according to the
        pixel_scale (determined by the sampling and the FOV) as well as the
        zero padding factor"""
        return fftshift(fftfreq(self.zero_padding, self.pixel_scale))

    @property
    def fftscaling2m(self):
        """Fetches the FFT's scaling in meters"""
        return np.diff(self.freq_axis)[0]/mas2rad()*self.wl

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
        return np.linspace(-self.fftaxis_m_end, self.fftaxis_m_end,
                           self.dim, endpoint=False)

    @property
    def fftaxis_Mlambda(self):
        """Gets the FFT's axis's endpoints in mega lambdas"""
        return np.linspace(-self.fftaxis_Mlambda_end, self.fftaxis_Mlambda_end,
                           self.dim, endpoint=False)

    @property
    def zero_padding(self):
        """The new pixel size to be had after zero padding"""
        return 2**int(np.log2(self.model_unpadded_dim)+self.zero_padding_order)

    def zero_pad_model(self):
        """This adds zero padding to the model image before it is transformed
        to increase the sampling in the FFT image

        As model image is in format of a power of 2 + 1 to ensure a centre
        pixel this shows in the code
        """
        padded_image = np.zeros((self.zero_padding, self.zero_padding))
        self.pad_centre = padded_image.shape[0]//2
        self.mod_min, self.mod_max = self.pad_centre-self.model_centre,\
                self.pad_centre+self.model_centre

        padded_image[self.mod_min:self.mod_max,
                     self.mod_min:self.mod_max] = self.model
        return padded_image

    def get_uv2fft2(self, uvcoords: np.ndarray, uvcoords_cphase: np.ndarray,
                intp: Optional[bool] = True, corr_flux: Optional[bool] = False,
                vis2: Optional[bool] = False) -> np.ndarray:
        """Interpolate the uvcoordinates to the grid of the FFT

        Parameters
        ----------
        uvcoords: np.ndarray
            The uv-coords for the correlated fluxes
        uvcoords_cphase: np.ndarray
            The uv-coords for the closure phases
        intp: bool, optional
            Decides if coordinates are interpolated on the grid or rounded
        corr_flux: bool, optional
            If the input image is a temperature gradient model then set this to
            'True' and the output will be the correlated fluxes
        vis2: bool, optional
            Will take the complex conjugate if toggled. Only works if
            'corr_flux' is 'False'

        Returns
        -------
        amp: np.ndarray
            The interpolated amplitudes
        cphases: np.ndarray
            The interpolated closure phases
        xy_coords: List
            This list contains the indices for overplotting the uv-coordinates
            of the visibilitites/correlated fluxes as well as the
            uv-coordinates for the closure phases to overplot on the phase plot
        """
        if intp:
            grid = (self.fftaxis_m, self.fftaxis_m)

            real_corr = interpn(grid, np.real(self.ft), uvcoords,
                                method='linear', bounds_error=False,
                                fill_value=None)
            imag_corr = interpn(grid, np.imag(self.ft), uvcoords,
                                method='linear', bounds_error=False,
                                fill_value=None)

            amp = np.abs(real_corr + 1j*imag_corr)

            real_phase = interpn(grid, np.real(self.ft), uvcoords_cphase,
                                  method='linear', bounds_error=False,
                                  fill_value=None)
            imag_phase = interpn(grid, np.imag(self.ft), uvcoords_cphase,
                                  method='linear', bounds_error=False,
                                  fill_value=None)
            cphases = np.angle(real_phase + 1j*imag_phase, deg=True)

            u_c, v_c = uvcoords_cphase
            xy_c = []
            for i, o in enumerate(u_c):
                for x, y in zip(o, v_c[i]):
                    xy_c.append([x, y])

            xy_coords = [np.flip(uvcoords, axis=1), np.array(xy_c)]

        else:
            xy_vis = []
            for uv in uvcoords:
                x, y = map(lambda x: self.model_centre +\
                           np.round(x/self.fftscaling2m).astype(int), uv)
                xy_vis.append([y, x])

            xy_vis = np.array(xy_vis)
            amp = np.abs(self.ft)[xy_vis]

            xy_cphase = []
            x_c, y_c = map(lambda x: self.model_centre +\
                       np.round(x/self.fftscaling2m).astype(int), uvcoords_cphase)

            for i, o in enumerate(y_c):
                for x, y in zip(x_c[i], o):
                    xy_cphase.append([y, x])

            xy_cphase = np.array(xy_cphase)
            cphases = np.angle(self.ft, deg=True)[xy_cphase]

            xy_coords = [np.flip(xy_vis, axis=1), np.flip(xy_cphase, axis=1)]

        cphases = sum(cphases)
        cphases = np.degrees((np.radians(cphases) + np.pi) % (2*np.pi) - np.pi)

        if not corr_flux:
            amp /= np.abs(self.ft_centre)
            if vis2:
                amp = amp*np.conjugate(amp)

        return amp, cphases, xy_coords

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
            amp, phase  = np.abs(self.ft)/np.abs(self.ft_centre),\
                    np.angle(self.ft, deg=True)

        return amp, phase

    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT and shifts the centre to the
        middle

        Returns
        --------
        ft: np.ndarray
        """
        self.model = self.zero_pad_model()
        self.raw_fft = fft2(ifftshift(self.model))
        self.ft_centre = self.raw_fft[0][0]
        return fftshift(self.raw_fft)


    def plot_amp_phase(self, matplot_axis: Optional[List] = [],
                       zoom: Optional[int] = 500,
                       corr_flux: Optional[bool] = True,
                       uvcoords_lst: Optional[List] = [],
                       plt_save: Optional[bool] = False) -> np.ndarray:
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
        uvcoords_lst: List, optional
            If not empty then the interpolation will be overplotted with the
            given (u,v)-coordinates
        plt_save: bool, optional
            Saves the plot if toggled on


        Returns
        -------
        np.ndarray
            The zoomed axis of the plot
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
        ax.imshow(self.unpadded_model, vmax=vmax, interpolation="None",
                  extent=[-fov_scale, fov_scale, -fov_scale, fov_scale])
        cbx = bx.imshow(amp, extent=[-self.fftaxis_m_end,
                                     self.fftaxis_m_end-1,
                                     -self.fftaxis_Mlambda_end,
                                     self.fftaxis_Mlambda_end-1],
                        interpolation="None", aspect=self.wl*1e6)
        ccx = cx.imshow(phase, extent=[-self.fftaxis_m_end,
                                       self.fftaxis_m_end-1,
                                       -self.fftaxis_Mlambda_end,
                                       self.fftaxis_Mlambda_end-1],
                        interpolation="None", aspect=self.wl*1e6)


        label_vis = "Flux [Jy]" if corr_flux else "vis"
        fig.colorbar(cbx, fraction=0.046, pad=0.04, ax=bx, label=label_vis)
        fig.colorbar(ccx, fraction=0.046, pad=0.04, ax=cx, label="Phase [°]")

        ax.set_title(f"Model image at {self.wl}, Object plane")
        bx.set_title("Amplitude of FFT")
        cx.set_title("Phase of FFT")

        ax.set_xlabel("RA [mas]")
        ax.set_ylabel("DEC [mas]")

        bx.set_xlabel("u [m]")
        bx.set_ylabel(r"v [M$\lambda$]")

        cx.set_xlabel("u [m]")
        cx.set_ylabel(r"v [M$\lambda$]")

        if uvcoords_lst:
            uvcoords, uvcoords_cphase = uvcoords_lst
            u, v = np.split(uvcoords, 2, axis=1)
            u_c, v_c = np.split(uvcoords_cphase, 2, axis=1)
            v, v_c = map(lambda x: x/(self.wl*1e6), [v, v_c])

            bx.scatter(u, v, color="r")
            cx.scatter(u_c, v_c, color="r")

        bx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])
        cx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])

        fig.tight_layout()

        if plt_save:
            plt.savefig(f"{self.wl}_FFT_plot.png")
        else:
            plt.show()


if __name__ == "__main__":
    ...
