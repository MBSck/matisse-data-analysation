import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.models import Gauss2D

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    file = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(file)
    wavelength = readout.get_wl()[80]
    uvcoords = readout.get_uvcoords()

    model = Gauss2D()
    model_vis  = model.eval_vis(128, 35.5124433, wavelength, uvcoords)
    print(model_vis)
    # plt.imshow(model_vis)
    # plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
