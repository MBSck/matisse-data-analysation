import numpy as np

from modelling import timeit, set_size, set_uvcoords, delta_fct

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
    flux: float
        The flux of the system
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
    def eval_model(self, size: int, major: int, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)

        try:
            return np.array([[(flux*delta_fct(j, major/2))/(np.pi*major) for j in i] for i in radius])
        except ZeroDivisionError:
            return np.array([[(flux*delta_fct(j, major/2))/(np.pi) for j in i] for i in radius])

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

