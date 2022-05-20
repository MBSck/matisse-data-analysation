import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, set_size, set_uvcoords,\
        mas2rad, get_px_scaling


class Binary(Model):
    """..."""
    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
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

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        ...


    def eval_vis(self, theta: np.ndarray, sampling: int,
                 wavelength: float, uvcoords: np.ndarray = None) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
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
            flux_ratio = theta[0]
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [flux_ratio]")

        numerator = 1+flux_ratio**2+2*flux_ratio*np.cos(2*np.pi/(wavelength*B*rho))
        denominator = (1+flux_ratio)**2
        return numerator/denominator
