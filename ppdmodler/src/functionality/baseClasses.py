import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional


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
        self._size, self._sampling = 0, 0
        self._wavelength = 0.
        self.flux, self.total_flux = [], 0.
        self._axis_mod, self._axis_vis = [], [], []

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


