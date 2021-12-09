import numpy as np
import matplotlib.pyplot as plt

from modelling.functionality.utilities import Model, timeit, set_size, set_uvcoords, \
        temperature_gradient, blackbody_spec

from modelling.models.ring import Ring

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
    def add_rings(self, size, min_radius: int, max_radius: int, step_size: int,
                  q: float, T_0: int, wavelength: float, do_flux: bool = True,
                  optical_depth: float = 1., inc_angle: int = 0, pos_angle_axis: int = 0,
                  pos_angle_ellipsis: int = 0, inclined: bool = False) -> None:
        """This adds the rings up to various models"""
        # TODO: Make this more performant -> Super slow

        output_lst = np.zeros((size, size))
        # Check if flux should be added to disk
        if do_flux:
            for i in range(min_radius+1, max_radius+2, step_size):
                # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)
                # Get the flux
                flux = blackbody_spec(i, q, min_radius, T_0, wavelength)*(1-np.exp(-optical_depth))
                # Get a single ring
                ring_array = Ring().eval_numerical(size, i, inc_angle=inc_angle, pos_angle_axis=pos_angle_axis, pos_angle_ellipsis=pos_angle_ellipsis, inclined=inclined)
                # Set the indices of the list to their value
                output_lst[np.where(ring_array > 0)] = flux/(np.pi*max_radius)
        else:
            for i in range(min_radius+1, max_radius+2, step_size):
                ring_array = Ring().eval_numerical(size, i, inc_angle=inc_angle, pos_angle_axis=pos_angle_axis, pos_angle_ellipsis=pos_angle_ellipsis, inclined=inclined)
                output_lst[np.where(ring_array > 0)] = 1/(np.pi*max_radius)

        return output_lst

    @timeit
    def uniformly_bright_disk(self, size: int, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(size, 0, radius, step_size, q, T_0, wavelength, do_flux=False)


    @timeit
    def optically_thin_disk(self, size: int, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(size, 0, radius, step_size, q, T_0, wavelength, optical_depth=0.1)

    @timeit
    def optically_thick_disk(self, size: int, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(size, 0, radius, step_size, q, T_0, wavelength, optical_depth=100.)


    @timeit
    def rimmed_disk(self, size: int, inner_radius: int, outer_radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1, do_flux: bool = True) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a disk with a inner ring

        See also
        --------
        add_rings()
        """
        return self.add_rings(size, inner_radius, outer_radius, step_size, q, T_0, wavelength, do_flux)

    @timeit
    def inclined_disk(self, size: int, inner_radius: int, outer_radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1, do_flux: bool = True) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a disk with a inner ring

        See also
        --------
        add_rings()
        """
        return self.add_rings(size, inner_radius, outer_radius, step_size, q, T_0, wavelength, inc_angle=60, pos_angle_axis=45, pos_angle_ellipsis=45, inclined=True)

if __name__ == "__main__":
    integ = IntegrateRings()
    plt.imshow(integ.inclined_disk(1024, 20, 50))
    plt.show()
    plt.imshow(integ.uniformly_bright_disk(1024, 50))
    plt.show()
    plt.imshow(integ.rimmed_disk(1024, 20, 50))
    plt.show()
    plt.imshow(integ.optically_thin_disk(1024, 50))
    plt.show()

