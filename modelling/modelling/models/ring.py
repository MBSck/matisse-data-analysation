import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords, delta_fct

class Ring(Model):
    """Infinitesimal thin ring model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    inc_angle: int
        The angle of the ring's i)nclination, defaults to 0.
    centre
        The centre of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, major: int, step: int = 1, centre: bool = None) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)

        try:
            return np.array([[delta_fct(j, major/2)/(np.pi*major) for j in i] for i in radius])
        except ZeroDivisionError:
            return np.array([[delta_fct(j, major/2)/(np.pi) for j in i] for i in radius])

    @timeit
    def eval_vis(self, major: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return j0(2*np.pi*major*B)

    @timeit
    def eval_numerical(self, size: int, outer_radius: int, inner_radius: int = None, inc_angle: int = 0, pos_angle_axis: int = 0, pos_angle_ellipsis: int = 0, centre: bool = None, inclined: bool = False) -> np.array:
        """Numerically evaluates the ring model"""
        x = np.arange(0, size)
        y = x[:, np.newaxis]
        inc_angle = np.radians(inc_angle)
        pos_angle_axis = np.radians(pos_angle_axis)
        pos_angle_measure = np.radians(pos_angle_ellipsis)

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

        return radius

if __name__ == "__main__":
    r = Ring()
    # for i in range(10, 90, 5):
    #     inclined_ring =  r.eval_numerical(512, 50, inc_angle=i, pos_angle_axis=45, pos_angle_ellipsis=45, inclined=True)
    #     plt.imshow(inclined_ring)
    #     plt.show()

    r_model = r.eval_model(512, 50)
    r_vis = r.eval_vis(0.1)
    plt.imshow(r_model)
    plt.show()
    plt.imshow(r_vis)
    plt.show()

