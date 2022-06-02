import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords, correspond_uv2model
from src.functionality.genetic_algorithm import genetic_algorithm, decode
from src.models import Gauss2D, CompoundModel
from src.functionality.fourier import FFT

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    readout = ReadoutFits(path)
    wavelength = 8e-6

    uvcoords = readout.get_uvcoords()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    cp = CompoundModel(1500, 7900, 19, 140, wavelength)

    cp_model, cp_flux = cp.eval_model([0.2, 45, 1., 1., 3., 0.04, 0.7], 10, 129)
    ax1.imshow(cp_flux)
    ax1.set_title("Temperature gradient")

    # TODO: Check the interpolation of this and plot it over the datapoints
    fft = FFT(cp_flux, wavelength, cp.pixel_scale, 4)
    amp, phase = fft.get_amp_phase(corr_flux=True)
    ax2.imshow(amp)
    ax2.set_title("FFT of Temperature gradient")

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
#    data = straigth((0.5, 2))[1]
#    print(decoded := test_chi_sq())
#    plt.plot(*straigth((0.5, 2)))
#    plt.plot(*straigth(decoded))
#    plt.show()

    main()

