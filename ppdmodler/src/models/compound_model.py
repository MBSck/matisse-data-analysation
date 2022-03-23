import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, mas2rad
from src.models import InclinedDisk, Delta

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
    def __init__(self ):
        super().__init__()
        self.name = "Compound-Model"
        self.r_outer = 0.
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
            param_lst = np.array(theta[3:])
            self.r_outer = mas2rad(theta[~0])
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [pos_angle_ellipsis, pos_angle_axis,"
                               " inc_angle, r_0, r_1, ...]")

        if sampling is None:
            sampling = size

        combined_model = np.zeros((sampling, sampling))
        combined_model += Delta().eval_model([1.], sampling)

        # TODO: Make list and dict that checks for the models and then uses
        # their specified eval_model functions as well as the right theta
        # slicing

        for i in range(0, len_param_lst := len(param_lst), 2):
            if i != len_param_lst-3:
                r_0, r_max = param_lst[i], param_lst[i+1]
                theta_mod = [r_0, r_max, pos_angle_ellipsis, pos_angle_axis, inc_angle]
                combined_model += InclinedDisk().eval_model(theta_mod, size,\
                                                          sampling, outer_r=self.r_outer)

        self._radius, self._axis_mod = set_size(size, sampling,\
                                                angles=[pos_angle_ellipsis, pos_angle_axis, inc_angle])
        self._radius_range = np.where(combined_model == 0.)
        # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)

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
    for i in range(0, 360, 10):
        cp_model = cp.eval_model([i, i, i, 0.17167066, 4.2379771], 128, 4096)
        plt.imshow(cp_model)
        plt.show()

