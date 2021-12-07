import numpy as np

from modelling.functionality.utilities import Model, timeit

class Delta(Model):
    """Delta function/Point source model

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    flux: float
        The flux of the system

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, step: int = 1, flux: float = 1.) -> np.array:
        """Evaluates the model

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.array([[0. for j in range(size)] if not i == size//2 else [0. if not j == size//2 else 1.*flux for j in range(size)] for i in range(size)])

    @timeit
    def eval_vis(self, size: int, flux: float = 1.) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        return np.ones((size, size))*flux

if __name__ == "__main__":
    delt = Delta()
    print(delt.eval_model(512))
