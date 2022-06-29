import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.functionality.utilities import trunc, azimuthal_modulation
from src.models import Gauss2D, Ring, CompoundModel, InclinedDisk, UniformDisk

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

def check_flux_behaviour(model, wavelength):
    """Plots the scaling behaviour of the fluxes"""
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

def check_interpolation(uvcoords, uvcoords_cphase, wavelength):
    u = CompoundModel(1500, 7900, 19, 140, wavelength)
    u_mod = u.eval_model([1.5, 135, 1, 1, 4., 0.4, 0.7], 10, 128)

    # Check previously calculated scaling factor
    print(10/128, u.pixel_scale)
    fft = FFT(u_mod, wavelength, 10/128, 5)
    amp_ip, cphase_ip, xy_coords_ip = fft.get_uv2fft2(uvcoords, uvcoords_cphase,
                                                      intp=True, corr_flux=True)
    amp, cphase, xy_coords = fft.get_uv2fft2(uvcoords, uvcoords_cphase,
                                             intp=False, corr_flux=True)
    fft.plot_amp_phase(corr_flux=True, uvcoords_lst=xy_coords,
                       plt_save=True)
    print("amps_inp", amp_ip, "amps", amp)
    print("cphases_inp", cphase_ip, "cphase", cphase)
    print("coords_inp", xy_coords_ip, "coords", xy_coords)

def main():
    wavelength = 10e-6
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    readout = ReadoutFits(path)

    uv = readout.get_uvcoords()
    uv_cphase = readout.get_t3phi_uvcoords()
    check_interpolation(uv, uv_cphase, wavelength)

if __name__ == "__main__":
    main()

