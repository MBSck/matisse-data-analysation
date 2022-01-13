import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from utilities import Model, timeit, set_size, set_uvcoords

class OpticallyThinSphere(Model):
    """Optically Thin Sphere model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, diameter: int, sampling: Optional[int] = None, flux: float = 1., centre: Optional[int] = None) -> np.array:
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
        model: np.array

        See also
        --------
        set_size()
        """
        diameter = np.radians(diameter/3.6e6)
        radius = set_size(size, sampling, centre)

        return np.array([[(6*flux/(np.pi*(diameter**2)))*np.sqrt(1-(2*j/diameter)**2) if j <= diameter/2 else 0 for j in i] for i in radius])

    @timeit
    def eval_vis(self, sampling: int, diameter: int, wavelength: float) -> np.array:
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
        B = set_uvcoords(sampling, wavelength)

        return (3/(np.pi*diameter*B)**3)*(np.sin(np.pi*diameter*B)-np.pi*diameter*B*np.cos(np.pi*diameter*B))

if __name__ == "__main__":
    o = OpticallyThinSphere()
    o_model = o.eval_model(512, 256.1)
    plt.imshow(o_model)
    plt.show()

    o_vis = o.eval_vis(512, 256.1, 8e-06)
    plt.imshow(o_vis)
    plt.show()

