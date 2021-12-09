import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords, \
        temperature_gradient, blackbody_spec

class InclinedDisk(Model):
    """By a certain position angle inclined disk

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    centre: bool
        The centre of the model, will be automatically set if not determined
    r_0: float
    T_0: float
    q: float
    inc_angle: float
    pos_angle_major: float
    pos_angle_measurement: float
    wavelength: float
    distance: float

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, q: float, r_0: float, T_0: int, r: float, wavelength: float, distance: int, inc_angle: int, step: int = 1, centre: bool = None) -> np.array:
        """Evaluates the Model

        Parameters
        ----------
        size: int
        q: float
        r_0: float
        T_0: float
        wavelength: float
        distance: float
        inc_angle: float
        step: int
        centre: bool

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)
        flux = blackbody_spec(radius, q, r_0, T_0, wavelength)

        # Sets the min radius r_0 and max radius r
        radius[radius < r_0] = 0
        radius[radius > r] = 0

        result = (2*np.pi*np.cos(inc_angle)*radius*flux)/distance
        print(result)

        return result

    @timeit
    def eval_vis(self, radius: float, q: float, r_0: float, T_0: int, wavelength: float, distance: int, inc_angle: int, pos_angle_axis: int, pos_angle_measure: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()
        # The ellipsis
        Bu, Bv = B*np.sin(pos_angle_measure), B*np.cos(pos_angle_measure)

        # Projected baselines with the rotation by the positional angle of the disk semi-major axis theta taken into account 
        Buth, Bvth = Bu*np.sin(pos_angle_axis)+Bv*np.cos(pos_angle_axis), \
                Bu*np.cos(pos_angle_axis)-Bv*np.sin(pos_angle_axis)
        B_proj = np.sqrt(Buth**2+Bvth**2*np.cos(inc_angle)**2)

        total_null_flux = self.eval_model(512, q, r_0, T_0, wavelength, distance, 0)

        return (1/total_null_flux)*blackbody_spec(radius, q, r_0, T_0, wavelength)*j0(2*np.pi*radius*B_proj)*(radius/distance)


if __name__ == "__main__":
    i = InclinedDisk()
    for inc in range(5, 90, 5):
        i_model = i.eval_model(size=512, q=0.55, r_0=100, r=150, T_0=6000, wavelength=8e-06, distance=1, inc_angle=inc)
        plt.imshow(i_model)
        plt.show()

    # i_vis = i.eval_vis(radius=0.25, q=0.55, r_0=0.01, T_0=6000, wavelength=8e-06, distance=1, inc_angle=60, pos_angle_axis=45, pos_angle_measure=45)
    # plt.imshow(i_vis)
    # plt.show()
