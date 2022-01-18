import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from src.functionality.utilities import Model, timeit, set_size, set_uvcoords, \
        temperature_gradient, blackbody_spec
from src.models.ring import Ring

class IntegrateRings:
    """Adds 2D rings up/integrates them to create new models (e.g., a uniform disk or a ring structure)

    ...

    Parameters
    ----------
    min_radius: int
    max_radius: int
    step_size: int
    q: float
    T_0: int
    wavelength: float

    Returns
    -------
    None

    Methods
    -------
    add_rings2D():
        This adds the rings up to various models and shapes
    uniformly_bright_disk():
        Calls the add_rings() function with the right parameters to create a uniform disk
    optically_thin_disk():
        Calls the add_rings() function with the right parameters to create a disk with an inner rim
    optically_thick_disk():
        ...
    rimmed_disk():
        ...
    """
    def integrate_rings(self, size: int, min_radius: int, max_radius: int, q: float, T_0: int, wavelength: float,
                        sampling: Optional[int] = None, optical_depth: float = 1., centre: Optional[int] = None) -> np.array:
        """This adds the ring's analytical model up to various models

        Parameters
        ----------

        Returns
        -------

        See also
        --------
        """
        # TODO: Make this more performant -> Super slow
        output_lst = np.zeros((size, size))

        if sampling is None:
            sampling = size

        for i in np.linspace(min_radius, max_radius):
            # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)
            flux = blackbody_spec(i, q, min_radius, T_0, wavelength)*(1-np.exp(-optical_depth))
            ring_array = Ring().eval_model(size, i, sampling, centre)*flux
            output_lst[np.where(ring_array > 0)] = flux/(np.pi*max_radius)

        return output_lst

    def integrate_rings_vis(self, sampling: int, min_radius: int, max_radius: int, q: float, T_0: int,
                            wavelength: float, optical_depth: float = 1.) -> np.array:
        """This adds the ring's analytical visibility up to various models

        Parameters
        ----------

        Returns
        -------

        See also
        --------
        """
        # TODO: Make this more performant -> Super slow
        output_lst = np.zeros((sampling, sampling)).astype(np.complex256)
        total_flux = 0

        for i in np.linspace(min_radius, max_radius):
            # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)
            visibility, flux = Ring().eval_vis(sampling, i, wavelength, True, min_radius, q, T_0)
            total_flux += flux
            ring_array =  visibility*flux*(1-np.exp(-optical_depth))
            output_lst += ring_array

        return output_lst/total_flux


if __name__ == "__main__":
    integ = IntegrateRings()

    # ax1.imshow(integ.integrate_rings(512, 1, 50, 0.55, 6000, 8e-06))
    plt.imshow(np.abs(integ.integrate_rings_vis(512, 1, 50, 0.55, 6000, 8e-06)))
    plt.savefig("integrate_rings_common_flux.png")
    plt.show()
