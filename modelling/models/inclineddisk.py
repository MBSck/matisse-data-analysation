class InclinedDisk(Model):
    """By a certain position angle inclined disk

    ...

    Attributes
    ----------
    size: float
        The size of the array that defines x, y-axis and constitutes the radius
    major: float
        The major determines the radius/cutoff of the model
    step: float
        The stepsize for the np.array that constitutes the x, y-axis
    centre: bool
        The centre of the model, will be automatically set if not determined
    r_0: float
        
    T_0: float
        
    q: float
        
    inc_angle: float
        
    pos_angle_major: float
        
    pos_angle_measurement: float
        
    wavelength: float
        
    distance: float
        

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    @timeit
    def eval_model(self, size: int, q: float, r_0: int, T_0: int, wavelength: float, distance: int, inclination_angle: int, step: int = 1, centre: bool = None) -> np.array:
        """Evaluates the Model

        Parameters
        ----------
        size: int
        q: float
        r_0: float
        T_0: float
        wavelength: float
        distance: float
        inc_angle: float
        step: int
        centre: bool

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        # Temperature gradient should be like 0.66, 0.65 or sth
        radius = set_size(size, step, centre)
        flux = blackbody_spec(radius, q, r_0, T_0, wavelength)
        factor = (2*np.pi/distance)*np.cos(inclination_angle)

        # Sets the min radius r_0
        radius[radius <= r_0] = 1

        try:
            result = factor*flux*radius
        except ZeroDivisionError:
            result = factor*flux

        return result

    @timeit
    def eval_vis(self) -> np.array:
        """Evaluates the visibilities of the model

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        self.B = set_uvcoords()
        self.Bu, self.Bv = self.B*np.sin(self.pos_angle_measure), self.B*np.cos(self.pos_angle_measure)     # Baselines projected according to their orientation
        self.Buth, self.Bvth = self.Bu*np.sin(self.pos_angle_major)+self.Bv*np.cos(self.pos_angle_major), \
                self.Bu*np.cos(self.pos_angle_major)-self.Bv*np.sin(self.pos_angle_major)                   # Baselines with the rotation by the positional angle of the disk semi-major axis theta taken into account 
        self.B_proj = np.sqrt(self.Buth**2+(self.Bvth**2)*np.cos(self.inc_angle)**2)                        # Projected Baseline


        return (1/self.eval_model(0))*self.blackbody_spec(radius)


