import numpy as np
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
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "CompoundModel"
        self.d, self.r = Delta(*args), Ring(*args)

    def get_flux(self, *args) -> np.array:
        flux = self.r.get_flux(*args)
        flux[self._size//2, self._size//2] = self.d.get_flux(*args, T_eff=7900)*1e26
        return flux

    @timeit
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
            r_0 = theta[0]
            ea, pa, inca = theta[1:]
        except:
            raise RuntimeError("Input of theta in wrong format")

        if sampling is None:
            sampling = px_size

        self._size = px_size

        image = self.r.eval_model([r_0, ea, pa, inca], mas_size, px_size, sampling)
        image *= azimuthal_modulation(self.r._phi, [[1, 1]])
        image += self.d.eval_model(px_size)

        return image

    def eval_vis():
        pass


if __name__ == "__main__":
    c = CompoundModel(1500, 7900, 19, 140, 8e-6)
    c_mod = c.eval_model([1.5, 45, 45, 45], 10, 128)
    c_flux = c.get_flux(0.5, 0.5, 1500, 19, 140, 8e-6)
    # print(c_flux)
    plt.imshow(c_flux)
    plt.show()
