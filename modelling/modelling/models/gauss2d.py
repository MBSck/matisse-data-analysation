import numpy as np
import matplotlib.pyplot as plt

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords

class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
    size: int
        The size of the array that defines x, y-axis and constitutes the radius
    major: int
        The major determines the radius/cutoff of the model
    step: int
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
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        radius = set_size(size, step, centre)
        return (flux/np.sqrt(np.pi/(4*np.log(2)*major)))*(np.exp(-4*np.log(2)*(radius**2)/(major**2)))

    @timeit
    def eval_vis(self, major: int) -> np.array:
        # TODO: Somehow relate the visibilites to the real actual model analytically -> Same major
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        B = set_uvcoords()

        return np.exp(-((np.pi*major*B)**2/(4*np.log(2))))

if __name__ == "__main__":
    g = Gauss2D()
    g_model = g.eval_model(0.01, 150)
    g_vis = g.eval_vis(0.01)
    plt.imshow(g_model)
    plt.show()
    plt.imshow(g_vis)
    plt.show()
