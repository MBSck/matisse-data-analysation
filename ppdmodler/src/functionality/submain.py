import numpy as np
import sys

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import interpolate, set_uvcoords

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    readout = ReadoutFits("/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits")
    uvcoords = readout.get_uvcoords()
    wavelength = 8e-06
    B, sampling, uv_ind, limits = interpolate(uvcoords, wavelength)
    B1 = set_uvcoords(sampling, wavelength, limits)
    print(limits, B1)

if __name__ == "__main__":
    main()
