import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.insert(1, os.path.abspath("../functionality"))

from utilities import Model, timeit

class Delta(Model):
    """Delta function/Point source model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, flux: float = 1.) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        size: int
            The size of the model image
        flux: float | None
            The flux of the object

        Returns
        --------
        model: np.array
        """
        output_array = np.zeros((size, size))
        output_array[size//2, size//2] = flux

        return output_array

    @timeit
    def eval_vis(self, sampling: int, flux: float = 1.) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        sampling: int
            The sampling of the uv-plane
        flux: float
            The flux of the object

        Returns
        -------
        visibility: np.array
        """
        return flux*np.ones((sampling, sampling))

if __name__ == "__main__":
    d = Delta()
    d_model = d.eval_model(512)
    plt.imshow(d_model)
    plt.show()

    d_vis = d.eval_vis(512)
    plt.imshow(d_vis)
    plt.show()

