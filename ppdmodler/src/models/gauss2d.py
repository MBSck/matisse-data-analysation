import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords, mas2rad

class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
        Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self):
        self.name = "2D-Gaussian"

    @timeit
    def eval_model(self, theta: List, size: int,
                   sampling: Optional[int] = None,  centre: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        fwhm: int | float
            The fwhm of the gaussian
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
            fwhm, flux = mas2rad(theta[0]), theta[1]
        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [fwhm, flux]")
            sys.exit()

        self._size, self._sampling = size, sampling
        radius, self._axis_mod  = set_size(size, sampling, centre)

        return (1/np.sqrt(np.pi/(4*np.log(2)*fwhm)))*np.exp((-4*np.log(2)*radius**2)/fwhm**2)

    @timeit
    def eval_vis(self, theta: np.ndarray, sampling: int,
                 wavelength: float, uvcoords: np.ndarray = None) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        fwhm: int | float
            The diameter of the sphere
        wavelength: int
            The sampling wavelength
        sampling: int, optional
            The sampling of the uv-plane
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
            fwhm = mas2rad(theta)
        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                      " the form [fwhm]")
            sys.exit()

        self._sampling = sampling
        B, self._axis_vis  = set_uvcoords(sampling, wavelength, uvcoords=uvcoords)

        return np.exp(-(np.pi*fwhm*B)**2/(4*np.log(2)))

if __name__ == "__main__":
    g = Gauss2D()
    g_model = g.eval_model([256.1, 1.], 512)
    plt.imshow(g_model)
    plt.show()

    g_vis = g.eval_vis(35, 512, 3.7e-06)
    plt.imshow(g_vis)
    plt.show()
