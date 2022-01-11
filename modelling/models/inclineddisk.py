import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0
from typing import Union, Optional

import os
import sys

sys.path.insert(1, os.path.abspath("../functionality"))

from utilities import Model, timeit, set_size, set_uvcoords, \
        temperature_gradient, blackbody_spec

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
        self.scaling = np.radians(1/3.6e6)

    @timeit
    def eval_model(self, size: int, r_max: Union[int, float], r_0: Union[int, float], T_0: int, q: float,
                   wavelength: float, distance: int, inc_angle: int, sampling: Optional[int] = None,
                   centre: Optional[int] = None) -> np.array:
        """Evaluates the Model

        Parameters
        ----------
        size: int
            The size of the model image
        r_max: int | float
            The cutoff radius of the inclined disk
        r_0: int | float
            The inital radius of the inclined disk
        T_0: int
            The temperature at the initial radius r_0
        q: float
            The temperature gradient
        wavelength: float
            The sampling wavelength
        distance: float
            The object's distance from the observer
        inc_angle: float
            The inclination angle of the disk
        sampling: int | None
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
        radius = set_size(size, sampling, centre)
        r_0, r_max = map(lambda x: x*self.scaling, [r_0, r_max])
        inc_angle = np.radians(inc_angle)

        flux = blackbody_spec(r_max, q, r_0, T_0, wavelength)

        output_lst = ((2*np.pi*radius*flux)/distance**2)*np.cos(inc_angle)
        output_lst[radius < r_0] = 0
        output_lst[radius > r_max] = 0

        return output_lst

    @timeit
    def eval_vis(self, sampling: int, radius: int, q: float, r_0: Union[int, float],
                 T_0: int, wavelength: float, distance: int, inc_angle: int,
                 pos_angle_axis: int, pos_angle_measure: int) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        sampling: int
            The sampling of the uv-plane
        radius: int
            The radius of the inclined disk
        r_0: int | float
            The inital radius of the inclined disk
        T_0: int
            The temperature at the initial radius r_0
        wavelength: float
            The sampling wavelength
        distance: int
            The object's distance from the observer
        inc_angle: int
            The inclination angle of the disk
        pos_angle_axis: int
        pos_angle_measurement: int

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        B = set_uvcoords(sampling, wavelength)

        # Convert angle to radians
        radius, r_0 =  map(lambda x: x*self.scaling, [radius, r_0])
        inc_angle, pos_angle_axis, pos_angle_measure = map(lambda x: np.radians(x), [inc_angle, pos_angle_axis, pos_angle_measure])

        # The ellipsis
        Bu, Bv = B*np.sin(pos_angle_measure), B*np.cos(pos_angle_measure)

        # Projected baselines with the rotation by the positional angle of the disk semi-major axis theta taken into account 
        Buth, Bvth = Bu*np.sin(pos_angle_axis)+Bv*np.cos(pos_angle_axis), \
                Bu*np.cos(pos_angle_axis)-Bv*np.sin(pos_angle_axis)
        B_proj = np.sqrt(Buth**2+Bvth**2*np.cos(inc_angle)**2)

        total_flux = self.eval_model(sampling, q, r_0, T_0, wavelength, distance, inc_angle=0)

        return (1/total_flux)*blackbody_spec(radius, q, r_0, T_0, wavelength)*j0(2*np.pi*radius*B_proj)*(radius/distance)


if __name__ == "__main__":
    i = InclinedDisk()
    for inc in range(5, 45, 5):
        i_model = i.eval_model(1024, 256.1, 10, 6000, 0.55, 8e-06, 1, inc)
        plt.imshow(i_model)
        plt.show()

    '''
    i_vis = i.eval_vis(radius=256.1, q=0.55, r_0=10, T_0=6000, sampling=512, wavelength=8e-06, distance=1, inc_angle=60, pos_angle_axis=45, pos_angle_measure=45)
    plt.imshow(i_vis)
    plt.show()
    '''
