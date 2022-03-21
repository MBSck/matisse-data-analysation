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
    wavelength = readout.get_wl()[80]

    # How to use corr_flux with vis in model
    # g = Gauss2D()
    # model = g.eval_model([1.], 128, 256)
    # flux = g.get_flux(0.55, 1500, 19, wavelength)

    cp = CompoundModel()
    model = cp.eval_model([45, 45, 45, 1., 3., 4., 6., 7., 10.], 128, 256)
    flux = cp.get_flux(0.55, 1500, 19, 140, wavelength)

    ft, amp, phase = FFT(model, wavelength).pipeline(vis2=True)

    amp *= np.sum(flux)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(amp)
    ax2.imshow(np.log(amp))
    plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
