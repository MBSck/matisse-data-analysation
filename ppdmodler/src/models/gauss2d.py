import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        mas2rad, get_px_scaling

from src.functionality.fourier import FFT

class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
        Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "2D-Gaussian"

    @timeit
    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        fwhm: int | float
            The fwhm of the gaussian
        q: float, optional
            The power law index
        size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane
        wavelength: float, optional
            The measurement wavelength
        centre: int, optional
            The centre of the model image

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            fwhm = theta[0]
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [fwhm]")
        if sampling is None:
            sampling = px_size

        self._size, self._sampling, self._mas_size = px_size, sampling, mas_size
        self._radius, self._axis_mod, self._phi  = set_size(mas_size, px_size, sampling)

        return (1/(np.sqrt(np.pi/(4*np.log(2)))*fwhm))*np.exp((-4*np.log(2)*self._radius**2)/fwhm**2)

    @timeit
    def eval_vis(self, theta: np.ndarray, sampling: int,
                 wavelength: float, uvcoords: np.ndarray = None) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        fwhm: int | float
            The diameter of the sphere
        wavelength: int
            The sampling wavelength
        sampling: int, optional
            The sampling of the uv-plane
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for
            precisely these.

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        try:
            fwhm = mas2rad(theta[0])
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be"
                               " of the form [fwhm]")

        self._sampling = sampling
        B, self._axis_vis  = set_uvcoords(sampling, wavelength=wavelength, uvcoords=uvcoords)

        return np.exp(-(np.pi*fwhm*B)**2/(4*np.log(2)))

if __name__ == "__main__":
    g = Gauss2D(1500, 7900, 19, 140, wavelength:=8e-6)
    g_model = g.eval_model([2.5], mas_fov:=10, sampling:=129)
    print(g.pixel_scale)
    g_flux = g.get_flux(np.inf, 0.7)
    g_tot_flux = g.get_total_flux(np.inf, 0.7)
    fig, (ax, bx, cx, dx) = plt.subplots(1, 4, figsize=(30, 5))
    fft = FFT(g_model, wavelength, g.pixel_scale, zero_padding_order=4)
    ft, amp2, phase = fft.pipeline()
    print(amp2[fft.model_unpadded_centre, fft.model_unpadded_centre], "0th element")
    ax.imshow(g_model, extent=[mas_fov, -mas_fov, -mas_fov, mas_fov])
    bx.imshow(g_flux, extent=[mas_fov, -mas_fov, -mas_fov, mas_fov])
    print(fft.fftaxis_m)
    ft_axis = fft.fftaxis_m
    ft_axis_Mlambda = fft.fftaxis_Mlambda
    cx.imshow(amp2, extent=[ft_axis, -ft_axis, -ft_axis, ft_axis])
    dx.imshow(amp2, extent=[ft_axis_Mlambda, -ft_axis_Mlambda,\
                           -ft_axis_Mlambda, ft_axis_Mlambda])

    ax.set_title("Model image, Object plane")
    bx.set_title("Temperature gradient")
    cx.set_title("Fourier transform of object plane (normed). vis")
    dx.set_title("Fourier transform of object plane (normed, zoomed). vis")

    ax.set_xlabel("RA [mas]")
    ax.set_ylabel("DEC [mas]")
    bx.set_xlabel("RA [mas]")
    bx.set_ylabel("DEC [mas]")
    cx.set_xlabel("u [m]")
    cx.set_ylabel("v [m]")
    dx.set_xlabel(r"u [M$\lambda$]")
    dx.set_ylabel(r"v [m$\lambda$]")
    plt.show()

