import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        mas2rad, blackbody_spec, sublimation_radius

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
                   sampling: Optional[int] = None, wavelength: float = None,
                   centre: Optional[int] = None,
                   bb_params: Optional[List[float]] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        fwhm: int | float
            The fwhm of the gaussian
        q: float, optional
            The power law index
        size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane
        wavelength: float, optional
            The measurement wavelength
        centre: int, optional
            The centre of the model image
        T_sub: int
            The sublimation temperature
        Luminosity_star: float
            The Luminosity of the star
        do_flux: bool, optional
            Calculates the flux if set to true

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            fwhm, flux = mas2rad(theta[0]), 1
            if len(theta) > 1:
                q = theta[1]

        except:
            print(f"{self.name}.{inspect.stack()[0][3]}(): Check input arguments, theta must be of"
                  " the form [fwhm] or [fwhm, q, T_0]")
            sys.exit()

        self._size, self._sampling = size, sampling
        radius, self._axis_mod  = set_size(size, sampling, centre)

        if bb_params is not None:
            T_sub, L_star = bb_params
            r_sub = sublimation_radius(T_sub, L_star)
            flux = blackbody_spec(radius, q, r_sub, T_sub, wavelength)

        radius *= mas2rad()

        return (flux/np.sqrt(np.pi/(4*np.log(2)*fwhm)))*np.exp((-4*np.log(2)*radius**2)/fwhm**2)

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
    g_model = g.eval_model([1., 0.55], 300, 128, wavelength=8e-06,
                           bb_params=[1500, 19])
    plt.imshow(g_model)
    plt.show()

    plt.show()
