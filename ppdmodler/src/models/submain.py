import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.functionality.utilities import trunc, azimuthal_modulation
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

def plot_all(model, mas_size, wavelength):
    mod = model.eval_model([35.57632429, 73.21246517, 81.30680296], mas_size, 1024)
    flux = model.get_flux(0.50409545, 0.3649148)
    tot_flux = model.get_total_flux(0.5, 0.55)
    print(tot_flux)

    fourier = FFT(mod, wavelength)
    ft, amp, phi = fourier.pipeline()
    corr_flux = amp*tot_flux

    u, v = (axis := np.linspace(-150, 150, 1024)), axis[:, np.newaxis]
    wavelength = trunc(wavelength*1e06, 2)
    xvis_curve = np.sqrt(u**2+v**2)[centre := 1024//2]
    yvis_curve = corr_flux[centre]

    fig, (ax, bx, cx, dx, ex) = plt.subplots(1, 5)
    ax.imshow(mod, extent=[mas_size/2, -mas_size/2, -mas_size/2, mas_size/2])
    bx.imshow(flux, extent=[mas_size/2, -mas_size/2, -mas_size/2, mas_size/2])
    cx.imshow(corr_flux, extent=[mas_size/2, -mas_size/2, -mas_size/2, mas_size/2])
    dx.plot(xvis_curve, yvis_curve)
    ex.imshow(np.log(amp*tot_flux), extent=[mas_size/2, -mas_size/2, -mas_size/2, mas_size/2])
    ax.set_title("Object Plane")
    bx.set_title("Temperature gradient")
    cx.set_title("Correlated Fluxes (vis*tot_flux)")
    dx.set_xlabel(r"$B_p$ [m]")
    ex.set_title("Correlated Fluxes (vis*tot_flux) [ln]")
    plt.show()

def main():
    wavelength = 9.5e-6

    # How to use corr_flux with vis in model
    c = CompoundModel(1500, 7900, 19, 140, wavelength)
    plot_all(c, 50, wavelength)

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()
