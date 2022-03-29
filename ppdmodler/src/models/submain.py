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
    for i in range(0, 100):
        for j in range(0, 5000, 100):
            j /= 1000
            print(j)
            wavelength = readout.get_wl()[i]
            print(wavelength)

            # How to use corr_flux with vis in model
            g = Gauss2D()
            model = g.eval_model([j], size:=128)
            flux = g.get_flux(j, 0.55, 1500, 19, 140, wavelength)
            print(flux[size//2, size//2], np.sum(flux))

            ft, amp, phase = FFT(model, wavelength).pipeline(vis2=False)
            print(amp[size//2, size//2])

            amp *= np.sum(flux)
            print(amp[size//2, size//2])
            print("---------------------------------")
        print("---------------------------------")
        print("---------------------------------")

    # plt.imshow(amp)
    # plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
