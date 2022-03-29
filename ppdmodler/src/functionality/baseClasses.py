import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

from src.functionality.utilities import plancks_law_nu, sublimation_radius,\
        sr2mas

# Classes

class Model(metaclass=ABCMeta):
    """Abstract metaclass that initiates the models

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self):
        self.name = ""
        self._radius, self._radius_range = [], []
        self._size, self._sampling = 0, 0
        self._axis_mod, self._axis_vis = [], []

    @property
    def size(self):
        return self._size

    @property
    def sampling(self):
        return self._sampling

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def axis_mod(self):
        return self._axis_mod

    def get_flux(self, obj_dim: float,
                 q: float, T_sub: int, L_star: float,
                 distance: float, wavelength: float) -> np.array:
        """Calculates the total flux of the model

        Parameters
        ----------
        obj_dim: float
            The size of the object
        q: float
            The power law index
        T_sub: int
            The sublimation temperature
        L_star: float
            The Luminosity of the star
        distance: float
            The distance to the object
        wavelength: float, optional
            The measurement wavelength

        Returns
        -------
        flux: np.ndarray
        """
        r_sub = sublimation_radius(T_sub, L_star, distance)
        flux = plancks_law_nu(self._radius, q, r_sub, T_sub, wavelength)

        if self._radius_range:
            flux[self._radius_range] = 0.

        return flux*sr2mas(obj_dim)*1e26

    @abstractmethod
    def eval_model() -> np.array:
        """Evaluates the model image
        Convention to put non fitting parameters at end of *args.

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass

    @abstractmethod
    def eval_vis() -> np.array:
        """Evaluates the visibilities of the model.
        Convention to put non fitting parameters at end of *args.

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass

@dataclass
class Parameter:
    """Class for keeping the parameters information"""
    name: str
    value: Union[int, float]

    def __call__(self):
        return self.value

    def __str__(self):
        return f"Param: {self.name} = {self.value}"

    def __repr__(self):
        return f"Param: {self.name} = {self.value}"

@dataclass
class Parameters:
    """A vector that gets all of the models parameters"""


