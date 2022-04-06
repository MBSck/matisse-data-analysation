import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from scipy.special import j0
from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.constant import I
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        delta_fct, mas2rad, trunc, azimuthal_modulation

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
                   inner_radius: Optional[int] = None) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Parameters
        ----------
        r_0: int | float
            The radius of the ring
        r_max: int | float, optional
            The radius if the disk if it is not None in theta
        pos_angle_ellipsis: int | float, optional
        pos_angle_axis: int | float, optional
        inc_angle: int | float, optional
            The inclination angle of the disk
        size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            pos_angle_ellipsis, pos_angle_axis, inc_angle = map(lambda x: np.radians(x), theta)
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [pos_angle_ellipsis,"
                               " pos_angle_axis, inc_angle]")

        if sampling is None:
            sampling = px_size

        self._size, self._sampling, self._mas_size = px_size, sampling, mas_size

        if len(theta) > 2:
            radius, self._axis_mod, self._phi = set_size(mas_size, px_size, sampling,
                                                  [pos_angle_ellipsis, pos_angle_axis, inc_angle])
        else:
            radius, self._axis_mod, self._phi = set_size(mas_size, px_size, sampling)

        if inner_radius:
            radius[radius < inner_radius] = 0.
        else:
            radius[radius < self._r_sub] = 0.

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
    r = Ring(1500, 7900, 19, 140, 8e-6)
    r_model = r.eval_model([4.53848935e+01, 4.47494126e+01, 4.57823954e+01], 100, 2048)
    print(r_flux := r.get_flux(2.64104748e-01, 3.70475743e-01))
    print(r.get_total_flux(0.5, 0.5))
    plt.imshow(r_flux)
    plt.show()

