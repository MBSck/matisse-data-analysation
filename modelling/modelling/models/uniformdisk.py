import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j1

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords, mas2rad


class UniformDisk(Model):
    """Uniformly bright disc model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius.
        The units are in radians.
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system
    RA
        The right ascension of the system
    DEC
        The declination of the system
    center
        The center of the model, will be automatically set if not determined

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, diameter: float, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        # Converts the mas to radians
        diameter = np.radians(diameter/3.6e6)

        output_lst = np.zeros((size, size))
        radius, theta =  set_size(size, step, centre)

        output_lst[radius <= diameter/2] = 4*flux/(np.pi*diameter**2)
        return output_lst

    @timeit
    def eval_vis(self, major: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()
        factor = np.pi*major*B

        return 2*j1(factor)/factor

if __name__ == "__main__":
    u = UniformDisk()
    u_model = u.eval_model(512, 251.6)
    # u_vis = u.eval_vis(0.1)
    plt.imshow(u_model)
    plt.show()
    # plt.imshow(u_vis)
    # plt.show()
