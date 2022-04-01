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
    model = g.eval_model([1], size:=128)
    for i in range(0,1000):
        i /= 100
        flux = g.get_flux(i, 0.5, 0.55, 1500, 19, 140, wavelength)
        print(i, ": ", flux)


if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
