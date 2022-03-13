import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit

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
    def __init__(self):
        self.name = "Delta"

    @timeit
    def eval_model(self, theta: List, size: int) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        flux: float | None
            The flux of the object
        size: int
            The size of the model image

        Returns
        --------
        model: np.array
        """
        try:
            flux = float(theta)
        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must"
                  " be of the form [flux]")
            sys.exit()

        self._size = size

        output_array = np.zeros((size, size))
        output_array[size//2, size//2] = flux

        return output_array

    @timeit
    def eval_vis(self, theta: List, sampling: int) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        flux: float
            The flux of the object
        sampling: int
            The sampling of the uv-plane

        Returns
        -------
        visibility: np.array
        """
        try:
            flux = float(theta)
        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must"
                  " be of the form [flux]")
            sys.exit()

        self._sampling = sampling

        return flux*np.ones((sampling, sampling))

if __name__ == "__main__":
    d = Delta()
    # d_model = d.eval_model([1, 10], 128)
    # plt.imshow(d_model)
    # plt.show()

    d_vis = d.eval_vis([1, 10], 128)
    plt.imshow(d_vis)
    plt.show()

