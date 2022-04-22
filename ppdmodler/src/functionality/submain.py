import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords, correspond_uv2model
from src.models import Gauss2D, CompoundModel
from src.functionality.fourier import FFT

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    readout = ReadoutFits(path)
    wavelength = 8e-6

    uvcoords = readout.get_uvcoords()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    cp = CompoundModel(1500, 7900, 19, 140, wavelength)

    cp_model = cp.eval_model([0.2, 45], 10, 129)
    cp_flux = cp.get_flux(np.inf, 0.7)
    ax1.imshow(cp_flux)
    ax1.set_title("Temperature gradient")

    # TODO: Check the interpolation of this and plot it over the datapoints
    fft = FFT(cp_flux, wavelength, cp.pixel_scale, 4)
    amp, phase = fft.get_amp_phase(True)
    print(amp[np.where(fft.interpolate_uv2fft2(uvcoords, True)[0])])
    ax2.imshow(amp)
    ax2.set_title("FFT of Temperature gradient")

    cp_tot_flux = cp.get_total_flux(np.inf, 0.7)
    fft = FFT(cp_model, wavelength, cp.pixel_scale, 4)
    amp2, phase2 = fft.get_amp_phase(False)
    ax3.imshow(amp2*cp_tot_flux)
    ax3.set_title("FFT of Object Plane * Total Flux")

    plt.show()

if __name__ == "__main__":
    main()

