class IntegrateRings:
    """Adds 2D rings up/integrates them to create new models (e.g., a uniform disk or a ring structure)

    ...

    Methods
    -------
    add_rings2D():
        This adds the rings up to various models and shapes
    uniform_disk():
        Calls the add_rings() function with the right parameters to create a uniform disk
    disk():
        Calls the add_rings() function with the right parameters to create a disk with an inner rim
    """
    def __init__(self, size_model: int) -> None:
        self.size = size_model

    def add_rings(self, min_radius: int, max_radius: int, step_size: int, q: float, T_0: int, wavelength: float, do_flux: bool = True, optical_depth: float = 1., optically_thick: bool = False) -> None:
        """This adds the rings up to various models

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
        """
        # TODO: Make this more performant -> Super slow
        output_lst = np.zeros((self.size, self.size))

        # Check if flux should be added to disk
        if do_flux:
            for i in range(min_radius+1, max_radius+2, step_size):
                # Optically thick (t_v >> 1 -> e^(-t_v)=0, else optically thin)
                if optically_thick:
                    flux = blackbody_spec(i, q, min_radius, T_0, wavelength)
                else:
                    flux = blackbody_spec(i, q, min_radius, T_0, wavelength)*(1-np.exp(-optical_depth))

                ring_array = Ring().eval_model(self.size, i)
                output_lst[np.where(ring_array > 0)] = flux/(np.pi*max_radius)
        else:
            for i in range(min_radius+1, max_radius+2, step_size):
                ring_array = Ring().eval_model(self.size, i)
                output_lst[np.where(ring_array > 0)] = 1/(np.pi*max_radius)

        return output_lst

    @timeit
    def uniformly_bright_disk(self, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(0, radius, step_size, q, T_0, wavelength, do_flux=False, optical_depth=1., optically_thick=False)


    @timeit
    def optically_thin_disk(self, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(0, radius, step_size, q, T_0, wavelength, do_flux=True, optical_depth=0.1, optically_thick=False)

    @timeit
    def optically_thick_disk(self, radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a uniform disk

        See also
        --------
        add_rings()
        """
        return self.add_rings(0, radius, step_size, q, T_0, wavelength, do_flux=True,  optical_depth=100., optically_thick=True)


    @timeit
    def rimmed_disk(self, inner_radius: int, outer_radius: int, wavelength: float = 8e-06, q: float = 0.55, T_0: float = 6000, step_size: int = 1, do_flux: bool = True, optical_depth: float = 0.01, optically_thick: bool = False) -> np.array:
        """Calls the add_rings2D() function with the right parameters to create a disk with a inner ring

        See also
        --------
        add_rings()
        """
        return self.add_rings(inner_radius, outer_radius, step_size, q, T_0, wavelength, do_flux, optical_depth, optically_thick)


