import numpy as np
import inspect
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, azimuthal_modulation, mas2rad
from src.models import Delta, Ring


class CompoundModel(Model):
    """Infinitesimal thin ring model. Can be both cirular or an ellipsoid, i.e.
    inclined

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model

    See also
    --------
    set_size()
    set_uvcoords()
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "Compound Model"
        self.d, self.r = Delta(*args), Ring(*args)

    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None) -> np.ndarray:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            if len(theta) < 4:
                axis_ratio, pa, c, s, tau, q = theta
            else:
                axis_ratio, pa, c, s, ring_inner_radius,\
                        ring_outer_radius, max_radius, tau, q = theta
                ring_outer_radius += ring_inner_radius
                max_radius += ring_outer_radius
            self.amplitudes = [[c, s]]
        except:
            raise RuntimeError(f"{self.name}.{inspect.stack()[0][3]}():"
                               " Check input arguments, theta must be of"
                               " the form [axis_ratio, pos_angle, c, s,"
                               " ring_inner_radius, ring_outer_radius,"
                               " inner_radius]")

        if sampling is None:
            self._sampling = sampling = px_size

        self._size, self._mas_size = px_size, mas_size

        image = self.r.eval_model([axis_ratio, pa], mas_size, px_size,
                                   sampling, inner_radius=max_radius)
        flux = self.r.get_flux(tau, q)
        temp_flux = flux.copy()

        self._max_sub_flux = np.max(flux)

        self.r._radius = None
        image += self.r.eval_model([axis_ratio, pa], mas_size, px_size,
                                  sampling, inner_radius=ring_inner_radius,
                                  outer_radius=ring_outer_radius)
        flux += self.r.get_flux(tau, q)
        flux *= azimuthal_modulation(self.r._phi, self.amplitudes)

        flux[sampling//2, sampling//2] = self.d.stellar_flux
        self._max_obj = np.max(image)

        return image, flux

    def eval_vis():
        pass


if __name__ == "__main__":
    c = CompoundModel(1500, 7900, 19, 140, 8e-6)
    c_mod, c_flux = c.eval_model([5.96207646e-02, 1.79327163e+02, 1.99972039e+00, 9.66528092e-01,
 5.84245342e-01, 3.97704978e+00, 2.31893470e+00, 6.23039738e-02,
                                  7.89946005e-01], 20, 4097)
    print(np.sum(c_flux))
    plt.imshow(c_flux, vmax=c._max_sub_flux)
    plt.show()

