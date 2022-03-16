import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, mas2rad
from src.models import InclinedDisk

class CompoundModel(Model):
    """Adds 2D rings up/integrates them to create new models (e.g., a uniform
    disk or a ring structure with or without inclination)

    ...

    Methods
    -------
    integrate_rings():
        This adds the rings up to various models and shapes
        ...
    integrate_rings_vis():
        ...
    """
    def __init__(self):
        super().__init__()
        self.name = "Compound-Model"
        self.r_outer_edge = 0.
        self._radius_range = []

    def eval_model(self, theta: List[float], size: int,
                        sampling: Optional[int] = None, optical_depth: float = 1.,
                        centre: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        theta: List
            Contains the r_0_n and r_max_n of the individual rings. The first
            three entries are the angles of the inclined disks

        Returns
        -------

        See also
        --------
        """
        try:
            pos_angle_ellipsis, pos_angle_axis, inc_angle = theta[:3]
            radii_lst = mas2rad(np.array(theta[3:]))
            self.r_outer_edge = mas2rad(theta[~0])
        except Exception as e:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [pos_angle_ellipsis, pos_angle_axis, inc_angle,"
                  " r_0, r_1, ...]")
            print(e)
            sys.exit()

        combined_model = np.zeros((size, size))
        self._radius, self._axis_mod = set_size(size, sampling,\
                                                angles=[pos_angle_ellipsis, pos_angle_axis, inc_angle])

        for i in range(0, len_radii := len(radii_lst), 2):
            print(radii_lst)
            if i != len_radii-3:
                r_0, r_max = radii_lst[i], radii_lst[i+1]
                print(r_0, r_max)

        # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)
        # flux = blackbody_spec(i, q, min_radius, T_0, wavelength)*(1-np.exp(-optical_depth))
        # ring_array = Ring().eval_model(size, i, sampling, centre)
        # combined_model[np.where(ring_array > 0)] = 1/(np.pi*self.r_outer_edge)

        return combined_model

    def eval_vis(self, sampling: int, min_radius: int, max_radius: int, q: float, T_0: int,
                            wavelength: float, optical_depth: float = 1.) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        wavelength: float
            The sampling wavelength
        sampling: int
            The sampling of the uv-plane

        Returns
        -------

        See also
        --------
        """
        output_lst = np.zeros((sampling, sampling)).astype(np.complex256)
        total_flux = 0

        for i in np.linspace(min_radius, max_radius):
            # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)
            visibility, flux = Ring().eval_vis(sampling, i, wavelength, True, min_radius, q, T_0)
            total_flux += flux
            ring_array =  visibility*flux*(1-np.exp(-optical_depth))
            output_lst += ring_array

        return output_lst/total_flux


if __name__ == "__main__":
    cp = CompoundModel()
    cp_model = cp.eval_model([45, 45, 45, 2., 10., 20., 40., 60., 80.], 128, 256)
    # plt.imshow(cp_model)
    # plt.show()

