import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional

from src.functionality.utilities import Model, timeit, set_size, set_uvcoords

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
        radius = set_size(size, sampling, centre)

        return (1/np.sqrt(np.pi/(4*np.log(2)*fwhm)))*np.exp((-4*np.log(2)*radius**2)/fwhm**2)

    @timeit
    def eval_vis(self, sampling: int, fwhm: Union[int, float], wavelength: float) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        sampling: int
            The sampling of the uv-plane
        fwhm: int | float
            The diameter of the sphere
        wavelength: int
            The sampling wavelength

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        B = set_uvcoords(sampling, wavelength)
        fwhm = np.radians(fwhm/3.6e6)

        return np.exp(-(np.pi*fwhm*B)**2/(4*np.log(2)))

if __name__ == "__main__":
    g = Gauss2D()
    # g_model = g.eval_model(512, 256.1)
    # plt.imshow(g_model)
    # plt.show()

    # TODO: Make scaling factor of px, the rest is already calculated to the
    # right distance/unit

    g_vis = g.eval_vis(512, 256.1, 8e-06)
    print(g_vis)
    plt.imshow(g_vis)
    plt.show()
