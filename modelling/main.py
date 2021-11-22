#!/usr/bin/env python3

__author__ = "Marten Scheuck"

from fourier import FFT
from modelling import ring2d, gauss2d, uniform_disk, optically_thin_sphere 

from utilities import ReadoutFits

def main():
    """Runs all of the main functionality"""
    # Calls the FFT, gets the model and gets  uv-coords from '.fits/.oifits'-file and finally does the FFT and gives its coords
    fourier = FFT(gauss2d(1000, 1500), "TARGET_CAL_INT_0001bcd_calibratedTEST.fits")
    print(fourier.get_fft_values(fourier.correspond_fft_to_freq()))
    fourier.do_plot(fourier.do_fft2(), fourier.correspond_fft_to_freq())

if __name__ == "__main__":
    main()
