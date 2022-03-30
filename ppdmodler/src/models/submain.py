import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.models import Gauss2D, Ring, CompoundModel, InclinedDisk

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    file = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(file)
    wavelength = readout.get_wl()[15]
    print(wavelength)

    # How to use corr_flux with vis in model
    g = Gauss2D()
    model = g.eval_model([3], size:=128)
    flux = g.get_flux(3, 0.55, 1500, 19, 140, wavelength)

    ft, amp, phase = FFT(model, wavelength).pipeline(vis2=False)
    print(amp[size//2, size//2])

    amp *= np.sum(flux)
    print(amp[size//2, size//2])
    plt.imshow(amp)
    plt.show()

    # plt.imshow(amp)
    # plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
