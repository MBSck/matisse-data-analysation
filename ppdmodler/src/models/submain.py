import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.functionality.utilities import trunc, azimuthal_modulation, \
        get_px_scaling
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

def plot_all(model, mas_size, wavelength, sampling):
    mod = model.eval_model([1, 140], mas_size, sampling)
    flux = model.get_flux(np.inf, 1)
    tot_flux = model.get_total_flux(np.inf, 1)
    print(tot_flux, "total flux")

    fourier = FFT(mod, wavelength, pixelscale=1.)
    ft, amp, phi = fourier.pipeline()

    fr_scale = get_px_scaling(fourier.fftfreq, wavelength)
    print(fourier.fftfreq, "scaling")
    print(fr_scale, "fr_scale")
    corr_scale = fr_scale*sampling/2
    print(corr_scale, "u_min")
    u, v = (axis := np.linspace(-corr_scale, corr_scale, sampling)),\
            axis[:, np.newaxis]
    corr_flux = amp*tot_flux


    # u, v = (axis := np.linspace(-150, 150, sampling)), axis[:, np.newaxis]
    wavelength = trunc(wavelength*1e06, 2)
    xvis_curve = np.sqrt(u**2+v**2)[centre := sampling//2]
    yvis_curve = amp[centre]
    yvis_curve2 = corr_flux[centre]

    third_max = len(corr_flux)//2 - 3

    fig, (ax, bx, cx, dx, ex) = plt.subplots(1, 5)
    ax.imshow(mod, vmax=model._max_obj,
              extent=[mas_size/2, -mas_size/2, -mas_size/2, mas_size/2])
    bx.imshow(flux, vmax=model._max_sub_flux, extent=[mas_size/2, -mas_size/2, -mas_size/2, mas_size/2])
    cx.plot(xvis_curve, yvis_curve)
    dx.plot(xvis_curve, yvis_curve2)
    ex.imshow(corr_flux, vmax=corr_flux[third_max, third_max],
             extent=[corr_scale, -corr_scale, -corr_scale, corr_scale])

    ax.set_title("Object Plane")
    bx.set_title("Temperature gradient")
    cx.set_title("Visibilities")
    dx.set_title("Correlated Fluxes")
    ex.set_title("Correlated Fluxes")
    ax.set_xlabel("[mas]")
    ax.set_ylabel("[mas]")
    bx.set_xlabel("[mas]")
    bx.set_ylabel("[mas]")
    cx.set_xlabel(r"$B_p$ [m]")
    cx.set_ylabel(r"vis")
    dx.set_xlabel(r"$B_p$ [m]")
    dx.set_ylabel(r"Corr Flux [Jy]")
    ex.set_xlabel(r"$u$ [m]")
    ex.set_ylabel(r"$v$ [m]")
    plt.show()

def main():
    wavelength, pixel_size = 9.5e-6, 10
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    readout = ReadoutFits(path)
    uv = readout.get_uvcoords()
    r = Ring(1500, 7900, 19, 140, wavelength)
    r_model = r.eval_model([0.5, 140], pixel_size, 129)
    r_tot_flux = r.get_total_flux(np.inf, 0.7)
    r_flux = r.get_flux(np.inf, 0.7)
    fft = FFT(r_flux, wavelength, r.pixel_scale, zero_padding_order=4)
    ft = fft.pipeline()
    ft = fft.interpolate_uv2fft2(ft, uv)
    print(ft)

if __name__ == "__main__":
    main()

