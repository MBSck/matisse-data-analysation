#!/usr/bin/env python3

I=complex(0, 1)
PLANCK_CONST: float = 6.62607015e-34        # Planck's constant in SI [J/Hz]
SPEED_OF_LIGHT: float = 2.99792458e8        # Speed of light in SI [m/s]
BOLTZMAN_CONST: float = 1.380649e-23        # Boltzman's constant in SI [J/K] 
STEFAN_BOLTZMAN_CONST: float = 5.670374419e-8   # in [W/m^2T^2]

AU_M: float = 149597870700  # in [m]

SOLAR_LUMINOSITY: float = 3.828e26  # in [W]

PARSEC2M: float =  3.085678e16    # in [m]

if __name__ == "__main__":
    print(BOLTZMAN_CONST)
