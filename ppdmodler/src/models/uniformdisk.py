import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional
from scipy.special import j1

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, mas2rad


class UniformDisk(Model):
    """Uniformly bright disc model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self):
        super().__init__()
        self.name = "Uniform Disk"

    @timeit
    def eval_model(self, theta: List, size: int,
                   sampling: Optional[int] = None, centre: Optional[int] = None) -> np.ndarray:
        """Evaluates the model

        Parameters
        ----------
        diameter: int
            The diameter of the sphere
        size: int
            The size of the model image
        sampling: int | None
            The sampling of the object-plane
        centre: int | None
            The centre of the model image

        Returns
        --------
        model: np.array[float]

        See also
        --------
        set_size()
        """
        try:
            diameter = mas2rad(theta[0])
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [diameter]")

        self._size, self._sampling = size, sampling
        radius, self._axis_mod = set_size(size, sampling, centre)
        print(diameter/2, radius[size//2, size//2])

        self._radius = radius.copy()

        radius[radius > diameter/2] = 0.
        radius[radius != 0.] = 4/(np.pi*diameter**2)
        self._radius_range = np.where(radius != 0)

        return radius

    @timeit
    def eval_vis(self, theta: List, sampling: int, wavelength:
                 float, uvcoords: np.ndarray = None) -> np.ndarray:
        """Evaluates the visibilities of the model

        Parameters
        ---------
        diameter: int
            The diameter of the sphere
        sampling: int
            The sampling of the uv-plane
        wavelength: float
            The sampling wavelength
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for
            precisely these.

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
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [diameter]")

        self._sampling, self._wavelength = sampling, wavelength
        B, self._axis_vis = set_uvcoords(sampling, wavelength, uvcoords)

        return 2*j1(np.pi*diameter*B)/(np.pi*diameter*B)

if __name__ == "__main__":
    u = UniformDisk()

    u_model = u.eval_model([4], 256)
    u_flux = u.get_flux(0.5, 0.5, 5, 1500, 19, 140, 8e-6)
    print(u_flux)
    plt.imshow(u_model)
    plt.show()

