import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

from src.functionality.utilities import plancks_law_nu, sublimation_radius,\
        sr2mas, temperature_gradient, stellar_radius_pc


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
        self._size, self._sampling, self._mas_size = 0, 0, 0
        self._axis_mod, self._axis_vis = [], []
        self._phi = []

    def get_total_flux(self, *args) -> np.ndarray:
        """Sums up the flux from [Jy/px] to [Jy]"""
        return np.ma.masked_invalid(self.get_flux(*args)).sum()

    def get_flux(self, optical_thickness: float,
                 q: float, T_sub: int, L_star: float,
                 distance: float, wavelength: float,
                 inner_radius: Optional[float] = None,
                 T_eff: Optional[int] = None) -> np.array:
        """Calculates the total flux of the model

        Parameters
        ----------
        optical_thickness: float
            The optical thickness of the disk, value between 0-1, which 1 being
            a perfect black body
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
        inner_radius: float, optional
            This sets an inner radius, different from the sublimation radius in [mas]
        T: int, optional
            The effective temperature

        Returns
        -------
        flux: np.ndarray
        """
        if inner_radius:
            r_sub = inner_radius
        else:
            r_sub = sublimation_radius(T_sub, L_star, distance)

        if T_eff:
            spectral_rad = plancks_law_nu(T_eff, wavelength)
            flux = np.pi*(stellar_radius_pc(T_eff, L_star)/distance)**2*spectral_rad
        else:
            T = temperature_gradient(self._radius, r_sub, q, T_sub)
            flux = plancks_law_nu(T, wavelength)
            flux *= (1-np.exp(-optical_thickness))*sr2mas(self._mas_size, self._sampling)*1e26

        return flux

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
    init_value: Union[int, float]
    value: Union[Any, int, float]
    error: Union[Any, int, float]
    priors: List[float]
    label: str
    unit: str

    def __call__(self) -> Union[int, float]:
        """Returns the parameter value"""
        return self.value

    def __str__(self) -> str:
        return f"Param='{self.name}': {self.value}+-{self.error}"\
                f" range=[{', '.join([str(i) for i in self.priors])}]"

if __name__ == "__main__":
    p = Parameter("x", 1.6, None, None, [0, 1], "x", "mas")
    print(p)

