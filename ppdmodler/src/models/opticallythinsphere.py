import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, mas2rad

class OpticallyThinSphere(Model):
    """Optically Thin Sphere model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self):
        self.name = "Optically Thin Sphere"

    @timeit
    def eval_model(self, theta: List, flux: float, size: int, sampling: Optional[int] = None, centre: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        diameter: int
            The diameter of the sphere
        flux: float
            The flux of the object
        size: int
            The size of the model image
        sampling: int | None
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
        try:
            diameter = mas2rad(theta)
        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [diameter]")
            sys.exit()

        self._size, self._sampling = size, sampling
        self._radius, self._axis_mod = set_size(size, sampling, centre)

        return np.array([[(6*flux/(np.pi*(diameter**2)))*np.sqrt(1-(2*j/diameter)**2)\
               if j <= diameter/2 else 0 for j in i] for i in self._radius])

    @timeit
    def eval_vis(self, theta: List, sampling: int, wavelength: float) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ---------
        diameter: int
            The diameter of the sphere
        sampling: int
            The sampling of the uv-plane
        wavelength: float
            The sampling wavelength

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        try:
            diameter = mas2rad(theta)
        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [diameter]")
            sys.exit()

        self._sampling, self._wavelength = sampling, wavelength
        B, self._axis_vis = set_uvcoords(sampling, wavelength)

        return (3/(np.pi*diameter*B)**3)*(np.sin(np.pi*diameter*B)-np.pi*diameter*B*np.cos(np.pi*diameter*B))

if __name__ == "__main__":
    o = OpticallyThinSphere()
    o_model = o.eval_model(256.1, 1., 128)
    plt.imshow(o_model)
    plt.show()

    o_vis = o.eval_vis(256.1, 128, 8e-06)
    plt.imshow(o_vis)
    plt.show()

