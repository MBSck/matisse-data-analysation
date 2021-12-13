import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0
from typing import Union, Optional

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords, delta_fct

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
    @timeit
    def eval_model(self, size: int, r_0: Union[int, float], sampling: Optional[int] = None, centre: Optional[bool] = None) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Parameters
        ----------
        size: int
            The size of the model image
        r_0: int | float
            The radius of the ring
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
        # TODO: Ring is 2 pixels thick, reduce to one?
        output_lst = np.zeros((size, size))
        r_0_temp = np.around(r_0*np.radians(1/3.6e6), decimals=8)
        radius = np.around(set_size(size, sampling, centre), decimals=8)
        output_lst[radius == r_0_temp] = 1/(2*np.pi*np.radians(r_0/3.6e6))
        return output_lst

    @timeit
    def eval_vis(self, sampling: int, r_0: Union[int, float], wavelength: float) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        sampling: int
            The sampling of the uv-plane
        r_0: int | float
            The radius of the ring
        wavelength: float
            The sampling wavelength

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        r_0 *= np.radians(1/3.6e6)
        B = set_uvcoords(sampling, wavelength)

        return j0(2*np.pi*r_0*B)

    @timeit
    def eval_numerical(self, size: int, outer_radius: int, inner_radius: Optional[int] = None,
                       inc_angle: int = 0, pos_angle_axis: int = 0, pos_angle_ellipsis: int = 0,
                       sampling: Optional[int] = None, centre: bool = None, inclined: bool = False) -> np.array:
        """Numerically evaluates the ring model"""
        # TODO: Check and apply correct unit conversions here
        if (sampling is None) or (sampling < size):
            sampling = size

        x = np.linspace(0, size, sampling)
        y = x[:, np.newaxis]

        # scaling
        scaling = np.radians(1/3.6e6)

        # Set angles to radians
        if inner_radius is not None:
            inner_radius *= scaling
        inc_angle *= np.radians(1)
        pos_angle_axis *= np.radians(1)
        pos_angle_ellipsis *= np.radians(1)

        if centre is None:
            x0 = y0 = size//2
        else:
            x0, y0 = centre

        # Calculates the radius from the centre and adds rotation to it
        xc, yc = x-x0, y-y0

        if inclined:
            a, b = xc*np.sin(pos_angle_ellipsis), yc*np.cos(pos_angle_ellipsis)
            ar, br = a*np.sin(pos_angle_axis)+b*np.cos(pos_angle_axis), \
                    a*np.cos(pos_angle_axis)-b*np.sin(pos_angle_axis)

            radius = np.sqrt(ar**2+br**2*np.cos(inc_angle)**2)
        else:
            radius = np.sqrt(xc**2+yc**2)

        # Gets the boundaries of the resulting ellipsis
        radius[radius > outer_radius] = 0.
        if inner_radius is None:
            radius[radius < outer_radius-1] = 0.
        else:
            radius[radius < inner_radius] = 0.

        radius *= scaling

        return radius

if __name__ == "__main__":
    r = Ring()
    # for i in range(10, 90, 5):
    #     inclined_ring =  r.eval_numerical(512, 50, inc_angle=i, pos_angle_axis=45, pos_angle_ellipsis=45, inclined=True)
    #     plt.imshow(inclined_ring)
    #     plt.show()

    r_model = r.eval_model(512, 10)
    plt.imshow(r_model)
    plt.show()

    r_vis = r.eval_vis(512, 10, 8e-06)
    plt.imshow(r_vis)
    plt.show()

    r_num = r.eval_numerical(512, 10)
    plt.imshow(r_num)
    plt.show()

