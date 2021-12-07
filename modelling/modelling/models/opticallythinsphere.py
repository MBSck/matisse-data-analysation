import numpy as np

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords, 

class OpticallyThinSphere(Model):
    """Optically Thin Sphere model

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
    def eval_model(self, size: int, major: int, step: int = 1, flux: float = 1., centre: bool = None) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)

        return np.array([[(6*flux/(np.pi*(major**2)))*np.sqrt(1-(2*j/major)**2) if j <= major//2 else 0 for j in i] for i in radius])

    @timeit
    def eval_vis(self, major: int) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return (3/(np.pi*major*B)**3)*(np.sin(np.pi*major*B)-np.pi*major*B*np.cos(np.pi*major*B))


