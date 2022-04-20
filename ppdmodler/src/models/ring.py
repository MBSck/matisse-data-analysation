import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from scipy.special import j0
from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.constant import I
from src.functionality.fourier import FFT
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        delta_fct, mas2rad, trunc, azimuthal_modulation, get_px_scaling

# TODO: Make the addition of the visibilities work properly, think of OOP
# abilities

class Ring(Model):
    """Infinitesimal thin ring model. Can be both cirular or an ellipsoid, i.e.
    inclined

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model

    See also
    --------
    set_size()
    set_uvcoords()
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "Ring"

    @timeit
    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None,
                   inner_radius: Optional[int] = 0,
                   outer_radius: Optional[int] = 0) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        r_0: int | float
            The radius of the ring
        r_max: int | float, optional
            The radius if the disk if it is not None in theta
        axis_ratio: float, optional
        pos_angle: int | float, optional
        px_size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane
        inner_radius: int, optional
            A set inner radius overwriting the sublimation radius

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            axis_ratio, pos_angle = theta
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [axis_ratio, pos_angle]")

        self._inner_r = inner_radius
        if sampling is None:
            self._sampling = sampling = px_size
        else:
            self._sampling = sampling

        self._size, self._mas_size = px_size, mas_size
        radius, self._axis_mod, self._phi = set_size(mas_size, px_size,\
                                                     sampling,\
                                                     [axis_ratio, pos_angle])

        if inner_radius:
            radius[radius < inner_radius] = 0.
        else:
            radius[radius < self._r_sub] = 0.

        if outer_radius:
            radius[radius > outer_radius] = 0.

        self._radius = radius.copy()
        radius[np.where(radius != 0)] = 1/(2*np.pi*self._r_sub)
        return radius

    @timeit
    def eval_vis(self, theta: List, sampling: int, wavelength: float, uvcoords:
                 np.ndarray = None, do_flux: bool = False) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        r_0: int | float
            The minimum radius of the ring, input in mas
            mas
        r_max: int | float
            The radius of the ring,  input in mas
        q: float
            The temperature gradient
        T_0: int
            The temperature at the minimum radias
        wavelength: float
            The sampling wavelength
        sampling: int
            The sampling of the uv-plane
        do_flux: bool
            Parameter that determines if flux is added to the ring and also
            returned

        Returns
        -------
        visibility: np.array
            The visibilities
        flux: np.array
            The flux. Will only get returned if 'do_flux=True'

        See also
        --------
        set_uvcoords()
        """
        try:
            r_0, r_max = map(lambda x: mas2rad(x), theta[:2])
            q, T_0 = theta[2:]
        except Exception as e:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of the"
                               " form [r_0, r_max, q, T_0]")

        self._sampling, self._wavelength = sampling, wavelength
        B, self._axis_vis = set_uvcoords(sampling, wavelength, uvcoords)

        # Realtive brightness distribution
        rel_brightness = self.eval_model(sampling, r_max)*blackbody_spec(r_max, q, r_0, T_0, wavelength)

        visibility = j0(2*np.pi*r_max*B)
        # TODO: Make way to get the individual fluxes from this funciotn
        # TODO: Remember to add up for the individual x, y and not sum them up,
        # should solve the problem

        x, u = np.linspace(0, sampling, sampling), np.linspace(-150, 150, sampling)/wavelength
        y, v = x[:, np.newaxis], u[:, np.newaxis]/wavelength

        return visibility*blackbody_spec(r_max, q, r_0, T_0, wavelength)*np.exp(2*I*np.pi*(u*x+v*y)),\
                rel_brightness


if __name__ == "__main__":
    # NOTE: There is a discrepancy between the model and the ASPRO model of
    # about 10m, possibly due to numerical effects?
    r = Ring(1500, 7900, 19, 140, wavelength:=10e-6)
    r_model = r.eval_model([1, 140], mas_fov:=10, sampling:=2049,\
                           outer_radius=(width:=1.05)*r._r_sub)
    r_flux = r.get_flux(np.inf, 0.7)
    r_tot_flux = r.get_total_flux(np.inf, 0.7)
    fig, (ax, bx, cx, dx) = plt.subplots(1, 4, figsize=(20, 5))
    fft = FFT(r_model, wavelength, r.pixel_scale, zero_padding_order=4)
    ft = fft.pipeline()
    amp, phase = fft.get_amp_phase(ft)
    ft_scaling = get_px_scaling(fft.fftfreq, wavelength)
    ft_ax = fft.fftaxis_m
    print(ft_ax, "scaling of the FFT to meters")
    print(r._r_sub, "sublimation radius")
    print(r._r_sub*width-r._r_sub, "width of the ring in mas")
    ft_lambda = fft.fftaxis_Mlambda
    ax.imshow(r_model, extent=[mas_fov, -mas_fov, -mas_fov, mas_fov])
    bx.imshow(r_flux, extent=[mas_fov, -mas_fov, -mas_fov, mas_fov])
    cx.imshow(amp, extent=[ft_ax, -ft_ax, -ft_ax, ft_ax])
    dx.imshow(amp, extent=[ft_lambda, -ft_lambda, -ft_lambda, ft_lambda])

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

