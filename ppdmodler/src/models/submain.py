import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.functionality.utilities import trunc
from src.models import Gauss2D, Ring, CompoundModel, InclinedDisk

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

def check_flux_behaviour(model, wavelength):
    lst = []
    pot_lst = [2**i for i in range(1, 10, 1)][3:]
    print(pot_lst)
    for j in pot_lst:
        mod = model.eval_model([1], 10, j)
        flux = model.get_flux(0.5, 0.55, 1500, 19, 140, wavelength)
        lst.append([j, flux])

    for i, o in enumerate(lst):
        if i == len(lst)//2:
            break


        print("|| ", o[0], ": ", trunc(o[1], 3),
              " || ", lst[~i][0], ": ", trunc(lst[~i][1], 3), " ||")

def plot_all(model, wavelength):
    mod = model.eval_model([1], 10, 128)
    flux = model.get_flux(0.5, 0.55, 1500, 19, 140, wavelength)

    fourier = FFT(mod, wavelength)
    ft, amp, phi = fourier.pipeline()

    print(flux)
    plt.imshow(amp*flux)
    plt.show()

def main():
    wavelength = 9.5e-6

    # How to use corr_flux with vis in model
    g = Gauss2D()
    check_flux_behaviour(g, wavelength)


if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
