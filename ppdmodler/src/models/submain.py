import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.models import Gauss2D

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    file = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(file)
    wavelength = readout.get_wl()[80]

    # How to use corr_flux with vis in model
    g = Gauss2D()
    model = g.eval_model([1.], 128, 256, wavelength)
    flux = g.get_flux(wavelength, 0.55, 1500, 19)

    ft, amp, phase = FFT(model, wavelength).pipeline(vis=True)

    amp *= flux

    print(amp)
    plt.imshow(amp)
    plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
