import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0
from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.constant import I
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        delta_fct, blackbody_spec, mas2rad, trunc

# TODO: Rescale the sizes of the models so that e.g. 512 points are up to 10
# arcseconds
# TODO: Make analytical model only one space wide (Ring)
# TODO: Make the addition of the visibilities work properly, think of OOP
# abilities

class Ring(Model):
    """Infinitesimal thin ring model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self):
        self.name = "Ring"
        self._axis_mod, self._axis_mod_num, self._axis_vis = [], [], []

    @property
    def axis_mod(self):
        return self._axis_mod

    @property
    def axis_mod_num(self):
        return self._axis_mod_num

    @property
    def axis_vis(self):
        return self._axis_vis

    @timeit
    def eval_model(self, theta: List, size: int,
                   sampling: Optional[int] = None, centre: Optional[bool] = None) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Parameters
        ----------
        r_0: int | float
            The radius of the ring
        size: int
            The size of the model image
        sampling: int | None
            The sampling of the object-plane
        centre: int | None
            The centre of the model image

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            r_0 = mas2rad(theta)
        except:
            print("Ring.eval_mod(): Check input arguments, theta must be of the form [r_0]")
            sys.exit()

        # TODO: Ring is 2 pixels thick, reduce to one?
        output_lst = np.zeros((sampling, sampling))
        radius, self._axis_mod = set_size(size, sampling, centre)

        r_0_trunc, radius_trunc = map(lambda x: trunc(x, 9), [r_0, radius])
        print(r_0, radius)
        print(r_0_trunc, radius_trunc)

        output_lst[radius_trunc == r_0_trunc] = 1/(2*np.pi*r_0)

        return output_lst

    @timeit
    def eval_mod_num(self, theta:  List, size: int,
                     sampling: Optional[int] = None,
                     centre: bool = None) -> np.array:
        """Numerically evaluates the ring model
        Parameters
        ----------
        r_0: int | float, optional
            The minimum radius of the ring input in radians
        r_max: int | float
            The radius of the ring input in radians
        inc_angle: float
            The inclination angle of the ring
        pos_angle_axis: float
        pos_angle_ellipsis: float
        size: int
            The size of the model image
        sampling: int, optional
            The sampling of the uv-plane
        centre: int, optional
            The centre of the model image

        Returns
        -------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            r_0, r_max = map(lambda x: mas2rad(x), theta[:2])
            pos_angle_ellipsis, pos_angle_axis, inc_angle = map(lambda x: np.radians(x), theta[2:])
        except:
            print("Ring.eval_mod_num(): Check input arguments, theta must be of the form [r_0,"
                  " r_max, pos_angle_ellipsis, pos_angle_axis, inc_angle]")
            sys.exit()

        radius, self._axis_mod_num = set_size(size, sampling, centre,
                                              [pos_angle_ellipsis, pos_angle_axis, inc_angle])

        radius[radius > r_max] = 0.
        if inner_radius is None:
            radius[radius < outer_radius-1] = 0.
        else:
            radius[radius < inner_radius] = 0.

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
        except:
            print("Ring.eval_vis(): Check input arguments, theta must be of the form [r_0,"
                  " r_max, q, T_0]")
            sys.exit()

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
    # inclined_ring =  r.eval_mod_num([50., 60, 45, 45], 512)
    # plt.imshow(inclined_ring)
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2)

    r_model = r.eval_model(10, 20, 512)
    plt.imshow(r_model)

    # r_vis = r.eval_vis([10., 256.1, 0.55, 6000], 512, 8e-06, do_flux=False)
    # print(r_vis, r_vis.shape)
    # ax2.imshow(np.abs(r_vis[0]))
    plt.show()
    # plt.savefig("mode_ring.png")

    '''
    r_num = r.eval_numerical(1024, 10)
    plt.imshow(r_num)
    plt.show()
    '''

