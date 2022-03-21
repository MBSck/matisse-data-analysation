import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from scipy.special import j0
from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.constant import I
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        delta_fct, mas2rad, trunc

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
    def __init__(self):
        super().__init__()
        self.name = "Ring"

    @timeit
    def eval_model(self, theta: List, size: int,
                   sampling: Optional[int] = None,
                   centre: Optional[bool] = None) -> np.array:
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
            if len(theta) == 1:
                r_0 = mas2rad(theta[0])
            else:
                r_0 = mas2rad(theta[0])
                pos_angle_ellipsis, pos_angle_axis, inc_angle = map(lambda x: np.radians(x), theta[1:])
        except Exception as e:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [r_0], or [r_0, pos_angle_ellipsis, pos_angle_axis,"
                  " inc_angle]")
            print(e)
            sys.exit()

        self._size, self._sampling = size, sampling

        if len(theta) > 2:
            radius, self._axis_mod = set_size(size, sampling, centre,
                                                  [pos_angle_ellipsis, pos_angle_axis, inc_angle])
        else:
            radius, self._axis_mod = set_size(size, sampling, centre)

        self._radius = radius.copy()

        radius[radius > r_0+mas2rad(1.)], radius[radius < r_0] = 0., 0.
        self._radius_range = np.where(radius == 0)
        radius[self._radius_range] = 1/(2*np.pi*r_0)

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
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of the form [r_0,"
                  " r_max, q, T_0]")
            print(e)
            sys.exit()

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
    r = Ring()

    r_model = r.eval_model([1.], 128, 256)
    print(r_flux := r.get_flux(8e-6, 0.55, 1500, 19))
    plt.imshow(r_model)
    plt.show()

