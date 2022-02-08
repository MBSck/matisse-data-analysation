import sys
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords

# Shows the full np.arrays, takes ages to print the arrays
np.set_printoptions(threshold=sys.maxsize)

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
        self._axis_mod= []
        self._axis_vis= []

    @property
    def axis_mod(self):
        return self._axis_mod

    @property
    def axis_vis(self):
        return self._axis_vis

    @timeit
    def eval_model(self, size: int, fwhm: Union[int, float],
                   sampling: Optional[int] = None, flux: float = 1., centre: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        size: int
            The size of the model image
        fwhm: int | float
            The fwhm of the gaussian
        sampling: int | None
            The sampling of the object-plane
        flux: float
            The flux of the object
        centre: int | None
            The centre of the model image

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        fwhm = np.radians(fwhm/3.6e6)
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
        sampling: int, optional
            The sampling of the uv-plane
        wavelength: int
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
        fwhm = theta
        fwhm = np.radians(fwhm/3.6e6)
        B, self._axis_vis  = set_uvcoords(sampling, wavelength, uvcoords=uvcoords)

        return np.exp(-(np.pi*fwhm*B)**2/(4*np.log(2)))

if __name__ == "__main__":
    g = Gauss2D()
    # g_model = g.eval_model(512, 256.1)
    # plt.imshow(g_model)
    # plt.show()

    # TODO: Make scaling factor of px, the rest is already calculated to the
    # right distance/unit

    g_vis = g.eval_vis(512, 35, 3.7e-06)
    print(g_vis[250:260])
    plt.imshow(g_vis)
    plt.show()
