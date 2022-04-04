import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.functionality.utilities import trunc
from src.models import Gauss2D, Ring, CompoundModel, InclinedDisk

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

def check_flux_behaviour(Model):
    lst = []
    for j in range(0, 4000):
        j /= 100
        flux = g.get_flux(j, 0.5, 0.55, 1500, 19, 140, wavelength)
        lst.append([j, flux])

    for i, o in enumerate(lst):
        if i == len(lst)//2:
            break

        temp_o = str(o[0]) if len(str(o[0])) == 4 else str(o[0])+"0"
        temp_i = str(lst[~i][0]) if len(str(lst[~i][0])) == 4 else str(lst[~i][0])+"0"

        print("|| ", temp_o, ": ", trunc(o[1], 3),
              " || ", temp_i, ": ", trunc(lst[~i][1], 3), " ||")

def main():
    wavelength = 9.5e-6

    # How to use corr_flux with vis in model
    g = Gauss2D()
    model = g.eval_model([1], 10, 128)
    flux = g.get_flux(10, 0.5, 0.55, 1500, 19, 140, wavelength)

    fourier = FFT(model, wavelength)
    ft, amp, phi = fourier.pipeline()

    plt.imshow(amp*flux)
    plt.show()


if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
