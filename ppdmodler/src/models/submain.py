import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.models import Gauss2D, Ring, CompoundModel, InclinedDisk

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    wavelength = 9.5e-6
    print(wavelength)

    # How to use corr_flux with vis in model
    g = Gauss2D()
    model = g.eval_model([3], size:=512)
    flux = g.get_flux(5, 0.5, 0.55, 1500, 19, 140, wavelength)

    ft, amp, phase = FFT(model, wavelength).pipeline(vis2=False)
    print(flux)
    print(amp[size//2, size//2])

    amp *= flux
    print(amp[size//2, size//2])

    plt.imshow(amp)
    plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
