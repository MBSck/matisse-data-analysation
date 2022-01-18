import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional
from scipy.special import j1

from src.functionality.utilities import Model, timeit, set_size, set_uvcoords, mas2rad


class UniformDisk(Model):
    """Uniformly bright disc model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, diameter: Union[int, float],
                   sampling: Optional[int] = None, flux: float = 1., centre: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        size: int
            The size of the model image
        diameter: int
            The diameter of the sphere
        sampling: int | None
            The sampling of the object-plane
        flux: float
            The flux of the object
        centre: int | None
            The centre of the model image

        Returns
        --------
        model: np.array[float]

        See also
        --------
        set_size()
        """
        # Converts the mas to radians
        diameter = np.radians(diameter/3.6e6)
        radius =  set_size(size, sampling, centre)

        output_lst = np.zeros((size, size))
        output_lst[radius <= diameter/2] = 4*flux/(np.pi*diameter**2)

        return output_lst

    @timeit
    def eval_vis(self, sampling: int, diameter: Union[int, float], wavelength: float) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ---------
        sampling: int
            The sampling of the uv-plane
        diameter: int
            The diameter of the sphere
        wavelength: float
            The sampling wavelength

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        diameter = np.radians(diameter/3.6e6)
        r = set_uvcoords(sampling, wavelength)

        return 2*j1(np.pi*diameter*r)/(np.pi*diameter*r)

if __name__ == "__main__":
    u = UniformDisk()
    u_model = u.eval_model(512, 256.1)
    plt.imshow(u_model)
    plt.show()

    u_vis = u.eval_vis(512, 256.1, 8e-06)
    plt.imshow(u_vis)
    plt.show()
