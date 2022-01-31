import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0
from typing import Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, \
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
                   wavelength: float, distance: int, inc_angle: int,
                   pos_angle_axis: int, pos_angle_measure: int, sampling: Optional[int] = None,
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
        if (sampling is None) or (sampling < size):
            sampling = size

        x = np.linspace(0, size, sampling)
        y = x[:, np.newaxis]

        if centre is None:
            x0 = y0 = size//2
        else:
            x0, y0 = centre

        inc_angle, pos_angle_axis, pos_angle_measure = map(lambda x: x*np.radians(1), [inc_angle, pos_angle_axis, pos_angle_measure])

        a, b = (x-x0)*np.sin(pos_angle_measure), (y-y0)*np.cos(pos_angle_measure)
        ar, br = a*np.sin(pos_angle_axis)+b*np.cos(pos_angle_axis), \
                a*np.cos(pos_angle_axis)-b*np.sin(pos_angle_axis)
        radius = np.sqrt(ar**2+br**2*np.cos(inc_angle)**2)*self.scaling

        r_0, r_max = map(lambda x: x*self.scaling, [r_0, r_max])

        output_lst = ((2*np.pi*radius*blackbody_spec(radius, q, r_0, T_0, wavelength))/distance)*np.cos(inc_angle)
        output_lst[radius < r_0], output_lst[radius > r_max] = 0, 0

        return output_lst

    @timeit
    def eval_vis(self, sampling: int, r_max: Union[int, float], q: float, r_0: Union[int, float],
                 T_0: int, wavelength: float, distance: int, inc_angle: int,
                 pos_angle_axis: int, pos_angle_measure: int) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        sampling: int
            The sampling of the uv-plane
        r_max: int | float
            The max radius of the inclined disk
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
        r_max, r_0 =  map(lambda x: x*self.scaling, [r_max, r_0])
        inc_angle, pos_angle_axis, pos_angle_measure = map(lambda x: np.radians(x), [inc_angle, pos_angle_axis, pos_angle_measure])

        # Sets the radius
        x = np.linspace(0, sampling, sampling)
        y = x[:, np.newaxis]

        x0 = y0 = sampling//2
        a, b = (x-x0)*np.sin(pos_angle_measure), (y-y0)*np.cos(pos_angle_measure)
        ar, br = a*np.sin(pos_angle_axis)+b*np.cos(pos_angle_axis), \
                a*np.cos(pos_angle_axis)-b*np.sin(pos_angle_axis)
        radius = np.sqrt(ar**2+br**2*np.cos(inc_angle)**2)*self.scaling

        # The ellipsis
        Bu, Bv = B*np.sin(pos_angle_measure), B*np.cos(pos_angle_measure)

        # Projected baselines with the rotation by the positional angle of the disk semi-major axis theta taken into account 
        Buth, Bvth = Bu*np.sin(pos_angle_axis)+Bv*np.cos(pos_angle_axis), \
                Bu*np.cos(pos_angle_axis)-Bv*np.sin(pos_angle_axis)
        B_proj = np.sqrt(Buth**2+Bvth**2*np.cos(inc_angle)**2)

        total_flux = self.eval_model(sampling, r_max, q, r_0, T_0, wavelength,
                                     distance, 0, pos_angle_axis,
                                     pos_angle_measure)

        output_lst = (1/total_flux)*blackbody_spec(radius, q, r_0, T_0, wavelength)*j0(2*np.pi*radius*B_proj)*(radius/distance)
        output_lst[radius < r_0], output_lst[radius > r_max] = 0, 0

        return output_lst


if __name__ == "__main__":
    i = InclinedDisk()
    for inc in range(10, 90, 10):
        i_model = i.eval_model(1024, 100, 20, 6000, 0.55, 8e-06, 1, inc, 45, 45)
        # i_vis = i.eval_vis(1024, 256.1, 0.55, 10, 6000, 8e-06, 1, inc, 45, 45)
        plt.imshow(i_model)
        # plt.imshow(i_vis)
        plt.show()

    '''
    i_vis = i.eval_vis(radius=256.1, q=0.55, r_0=10, T_0=6000, sampling=512, wavelength=8e-06, distance=1, inc_angle=60, pos_angle_axis=45, pos_angle_measure=45)
    plt.imshow(i_vis)
    plt.show()
    '''
