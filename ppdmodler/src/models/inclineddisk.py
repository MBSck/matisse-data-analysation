import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0
from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, \
        temperature_gradient, blackbody_spec, mas2rad

# TODO: Finish revamping both the eval_mod and eval_vis so the fit to the new
# data format
# TODO: Split the flux code from the other code and try to fit it. So that
# there is no wavelength dependendy in the eval.model() code

class InclinedDisk(Model):
    """By a certain position angle inclined disk

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """

    def __init__(self):
        super().__init__()
        self.name = "Inclined Disk"

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
            if len(theta) <= 2:
                r_0, r_max = map(lambda x: mas2rad(x), theta)
            else:
                r_0, r_max = map(lambda x: mas2rad(x), theta[:2])
                pos_angle_ellipsis, pos_angle_axis, inc_angle = map(lambda x: np.radians(x), theta[2:])
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

        radius[radius > r_max], radius[radius < r_0] = 0., 0.
        self._radius_range = np.where(radius == 0)
        radius[np.where(radius != 0)] = 1/(2*np.pi*r_0)

        return radius

#     @timeit
#     def eval_model(self, theta: List, wavelength: float,
#                    size: int, sampling: Optional[int] = None,
#                    centre: Optional[int] = None) -> np.array:
#         """Evaluates the Model
# 
#         Parameters
#         ----------
#         r_0: int | float
#             The inital radius of the inclined disk
#         r_max: int | float
#             The cutoff radius of the inclined disk
#         distance: float
#             The object's distance from the observer
#         q: float
#             The temperature gradient
#         T_0: int
#             The temperature at the initial radius r_0
#         pos_angle_ellipsis: float
#         pos_angle_axis: float
#         inc_angle: float
#             The inclination angle of the disk
#         wavelength: float
#             The sampling wavelength
#         size: int
#             The size of the model image
#         sampling: int, optional
#             The sampling of the object-plane
#         centre: int, optional
#             The centre of the model image
# 
#         Returns
#         --------
#         model: np.array
# 
#         See also
#         --------
#         set_size()
#         """
#         try:
#             r_0, r_max, distance  = map(lambda x: mas2rad(x), theta[:3])
#             q, T_0 = theta[3:5]
#             pos_angle_ellipsis, pos_angle_axis, inc_angle = map(lambda x: np.radians(x), theta[~2:])
#         except Exception as e:
#             print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
#                   " the form [r_0, r_max, distance, q, T_0,"
#                   " pos_angle_ellipsis, pos_angle_axis, inc_angle]")
#             print(e)
#             sys.exit()
# 
#         self._size, self._sampling, self._wavelength = size, sampling, wavelength
#         radius, self._axis_mod = set_size(size, sampling, angles=[pos_angle_ellipsis, pos_angle_axis, inc_angle])
# 
#         output_lst = ((2*np.pi*radius*blackbody_spec(radius, q, r_0, T_0, wavelength))/distance**2)*np.cos(inc_angle)
#         output_lst[radius < r_0], output_lst[radius > r_max] = 0., 0.
# 
#         return output_lst

    @timeit
    def eval_vis(self, theta: List, wavelength: float, sampling: int) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        r_max: int | float
            The max radius of the inclined disk
        r_0: int | float
            The inital radius of the inclined disk
        T_0: int
            The temperature at the initial radius r_0
        distance: int
            The object's distance from the observer
        pos_angle_ellipsis: float
        pos_angle_axis: float
        inc_angle: float
            The inclination angle of the disk
        wavelength: float
            The sampling wavelength
        sampling: int
            The sampling of the uv-plane

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        try:
            r_0, r_max, distance = map(lambda x: mas2rad(x), theta[:3])
            q, T_0 = theta[3:5]
            pos_angle_ellipsis, pos_angle_axis, inc_angle = map(lambda x: np.radians(x), theta[~2:])
        except Exception as e:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [r_0, r_max, q, T_0, pos_angle_ellipsis,"
                  " pos_angle_axis, inc_angle]")
            print(e)
            sys.exit()

        self._sampling, self._wavelength = sampling, wavelength

        # Sets the projected baseline for an ellipsis
        B, self._axis_vis = set_uvcoords(sampling, wavelength, angles=[pos_angle_ellipsis, pos_angle_axis, inc_angle])

        # Sets the radius for an ellipsis
        radius, temp = set_size(sampling, angles=[pos_angle_ellipsis, pos_angle_axis, inc_angle])

        # Set the inclination angle to 0 for theta_cp
        theta[~0] = 0.
        total_flux = self.eval_model(theta, wavelength, sampling)

        output_lst = (radius/total_flux)*blackbody_spec(radius, q, r_0, T_0, wavelength)*j0(2*np.pi*radius*B/distance)
        # output_lst[radius < r_0], output_lst[radius > r_max] = 0, 0

        return output_lst

if __name__ == "__main__":
    inclined = InclinedDisk()
    model = inclined.eval_model([3., 5., 45, 45, 45], 512)
    plt.imshow(model)
    plt.show()

    '''
    i_vis = i.eval_vis(radius=256.1, q=0.55, r_0=10, T_0=6000, sampling=512, wavelength=8e-06, distance=1, inc_angle=60, pos_angle_axis=45, pos_angle_measure=45)
    plt.imshow(i_vis)
    plt.show()
    '''
