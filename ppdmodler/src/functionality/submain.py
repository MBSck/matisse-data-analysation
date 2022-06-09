import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords
from src.functionality.genetic_algorithm import genetic_algorithm, decode
from src.models import Gauss2D, CompoundModel
from src.functionality.fourier import FFT

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    path = 
    readout = ReadoutFits(path)
    wavelength = 8e-6

    uvcoords = readout.get_uvcoords()

    plt.show()

def straigth(x):
    xcoords = np.linspace(0, 100)
    return xcoords, xcoords*x[0] + x[1]

def chi_sq(x):
    return np.sum((data - straigth(x)[1])**2/(data*0.2)**2)

def test_chi_sq():
    bounds = [[0., 1.], [0, 5]]
    n_bits = 32
    n_pop, n_iter = 100, 1000
    r_cross = 0.85
    best, scores = genetic_algorithm(chi_sq, bounds, n_bits, n_iter, n_pop,
                                     r_cross, 1./(n_bits*len(bounds)))
    decoded = decode(bounds, n_bits, best)
    return decoded

if __name__ == "__main__":
    data = straigth((0.5, 2))[1]
    print(decoded := test_chi_sq())
    plt.plot(*straigth((0.5, 2)), color="b")
    plt.plot(*straigth(decoded), color="r")
    plt.show()

#    main()

