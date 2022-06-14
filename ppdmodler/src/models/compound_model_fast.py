import numpy as np
import inspect
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.utilities import timeit, azimuthal_modulation, mas2rad
from src.models import Delta, Ring

from src.functionality.fourier import FFT

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
        self._axis_mod = []

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
            if len(theta) < 8:
                axis_ratio, pa, c, s, sub_radius, tau, q = theta
            else:
                axis_ratio, pa, c, s, mod_angle, ring_inner_radius,\
                        ring_outer_radius, tau, q = theta
            self.amplitudes = [[c, s]]
            ring_outer_radius += ring_inner_radius
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [axis_ratio, pos_angle, c, s,"
                          " ring_inner_radius, ring_outer_radius,"
                          " inner_radius]")

        if sampling is None:
            self._sampling = sampling = px_size

        self._size, self._mas_size = px_size, mas_size

        image = self.r.eval_model([axis_ratio, pa], mas_size, px_size,
                                  sampling, inner_radius=ring_inner_radius,
                                  outer_radius=ring_outer_radius)

        flux = self.r.get_flux(tau, q)
        self._max_sub_flux = np.max(flux)

        flux *= azimuthal_modulation(self.r._phi, mod_angle, self.amplitudes)
        flux[sampling//2, sampling//2] = self.d.stellar_flux

        return flux

    def eval_vis():
        pass


if __name__ == "__main__":
    wavelength, sampling, mas_size = 10e-6, 513, 50
    c = CompoundModel(1500, 7900, 19, 140, 8e-6)
    c_mod, c_flux = c.eval_model([0.6, 45, 1, 1, 4., 0.04, 0.7], mas_size,
                                 sampling)
    fft = FFT(c_flux, wavelength, c.pixel_scale, 3)
    amp, phase = fft.get_amp_phase()

    fft.plot_amp_phase()

