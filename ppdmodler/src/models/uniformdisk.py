import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional
from scipy.special import j1

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, mas2rad
from src.functionality.fourier import FFT


class UniformDisk(Model):
    """Uniformly bright disc model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "Uniform Disk"

    def eval_model(self, theta: List, mas_size: int,
                   px_size: int, sampling: Optional[int] = None) -> np.ndarray:
        """Evaluates the model

        Parameters
        ----------
        theta: List
            The list of the input parameters (diameter, axis_ratio, pos_angle)
        mas_size: int
            The field of view in mas
        px_size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane

        Returns
        --------
        model: np.ndarray

        See also
        --------
        set_size()
        """
        try:
            diameter, axis_ratio, pos_angle = theta
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [diameter, axis_ratio, pos_angle]")

        if not sampling:
            sampling = px_size

        self._size, self._sampling, self._mas_size = px_size, sampling, mas_size
        radius, self._axis_mod, self._phi = set_size(mas_size, px_size, sampling,
                                                     [axis_ratio, pos_angle])

        self._radius = radius.copy()

        radius[radius > diameter/2] = 0.
        radius[np.where(radius != 0)] = 4/(np.pi*diameter**2)
        self._radius_range = np.where(radius != 0)

        return radius

    def eval_vis(self, theta: List, sampling: int, wavelength:
                 float, uvcoords: np.ndarray = None) -> np.ndarray:
        """Evaluates the visibilities of the model

        Parameters
        ---------
        diameter: int
            The diameter of the sphere
        sampling: int
            The sampling of the uv-plane
        wavelength: float
            The sampling wavelength
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
            diameter = theta
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [diameter]")

        self._sampling, self._wavelength = sampling, wavelength
        B, self._axis_vis = set_uvcoords(sampling, wavelength, uvcoords)

        return 2*j1(np.pi*diameter*B)/(np.pi*diameter*B)

if __name__ == "__main__":
    wavelength, sampling, mas_fov  = 10e-6, 129, 10
    u = UniformDisk(1500, 7900, 19, 140, wavelength)

    u_model = u.eval_model([4, 1, 180], mas_fov, sampling)
    fft = FFT(u_model, wavelength, u.pixel_scale, 3)
    amp, phase = fft.get_amp_phase()

    ft_ax = fft.dim//2*fft.fftscaling2m
    ft_lambda = fft.fftaxis_Mlambda_end

    fig, axarr = plt.subplots(1, 3, figsize=(25, 7))
    ax, bx, cx = axarr.flatten()

    ax.imshow(u_model, extent=[-mas_fov, mas_fov, -mas_fov, mas_fov])
    bx.imshow(amp, extent=[-ft_ax, ft_ax, -ft_ax, ft_ax])
    cx.imshow(amp, extent=[-ft_lambda, ft_lambda, -ft_lambda, ft_lambda])

    ax.set_title("Model image, Object plane")
    bx.set_title("FFT of Model")
    cx.set_title("FFT of Model")

    ax.set_xlabel("RA [mas]")
    ax.set_ylabel("DEC [mas]")
    bx.set_xlabel("u [m]")
    bx.set_ylabel("v [m]")
    cx.set_xlabel(r"u [M$\lambda$]")
    cx.set_ylabel(r"v [M$\lambda$]")

    ax.axis([-5, 5, -5, 5])
    bx.axis([-300, 300, -300, 300])
    cx.axis([-70, 70, -70, 70])

    plt.show()

