#!/usr/bin/env python3

__author__ = "Marten Scheuck"

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import numpy as np                  # Imports the numpy module
import time                         # Imports the time module
import matplotlib.pyplot as plt     # Imports the matplotlib module

from numpy import fft               # Imports the Fast-Fourier Transform (FFT) pack
from pathlib import Path

# Own modules
from utilities import ReadoutFits, timeit
from gauss2d import Gauss2D

# Be aware of the fact that the code only works for same dimensional pictures/models

# TODO: Make class and function documentation

class FFT:
    """A collection of the fft-functionality given by scipy

    Attributes
    ----------

    """
    # TODO: Check how the indices are sorted, why do they change? They change even independent of the scaling
    # TODO: Remove staticmethods and make them outside of class
    # TODO: Change euclidean distance to interpolation in order to get coordinates
    # TODO: Make Zoom function work
    def __init__(self, model: np.array, set_size: int, fits_file_path: Path, wavelength: float, step_size_fft: float = 1., greyscale: bool = False) -> None:
        self.model= model                                                           # Evaluates the model
        self.model_size = len(self.model)                                           # Gets the size of the model's image

        # Initializes the readout for '.fits'-files
        self.readout = ReadoutFits(fits_file_path)


        # General variables
        self.set_size = set_size
        self.fftfreq = fft.fftfreq(self.model_size, d=step_size_fft)                # x-axis of the FFT corresponding to px/img_size
        self.min_freq, self.max_freq = np.min(self.fftfreq), np.max(self.fftfreq)   # x-axis lower and upper boundaries
        self.roll = np.floor(self.model_size/2).astype(int)                         # Gets half the pictures size as int
        self.freq = np.roll(self.fftfreq, self.roll, axis=0)                        # Rolls 0th-freq to centre
        self.fftscale = np.diff(self.freq)[0]                                       # cycles/mas per px in FFT img

        self.uvcoords = self.readout.get_uvcoords_vis2                              # Gets the uvcoords of vis2
        self.wavelength = wavelength                                                # The set wavelength for scaling

        # Conversion units
        self.mas2rad = np.deg2rad(1/3.6e6) # mas per rad

    def fft_pipeline(self) -> [float, np.array, float, float]:
        """A pipeline function that calls the functions of the FFT in order and
        avoids double calling of single functions"""
        ft = self.do_fft2()
        rescaling_factor = self.get_scaling_px2metr()
        rescaled_uvcoords = self.rescale_uvcoords(self.uvcoords)
        uv_ind = self.correspond_fft2freq(rescaled_uvcoords)
        # self.do_plot(ft, rescaled_uvcoords)
        fft_value = FFT.get_fft_values(ft, uv_ind)
        # ft = self.zoom_fft2(ft, self.set_size)
        return ft, rescaled_uvcoords, fft_value, FFT.get_ft_amp_phase(fft_value)

    @timeit
    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT"""
        return fft.fftshift(fft.fft2(fft.ifftshift(self.model)))

    @timeit
    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and returns the inverse 2D-FFT"""
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real

    def zoom_fft2(self, ft: np.array, set_size: int):
        """This zooms the FFT in after zero-padding"""
        ind_low_start, ind_low_end = 0, set_size//2
        ind_high_start, ind_high_end = set_size//2, self.set_size
        ft = np.delete(ft[:], np.arange(ind_low_start, ind_low_end, 0))
        ft = np.delete(ft[:], np.arange(ind_low_start, ind_low_end, 1))
        ft = np.delete(ft[:], np.arange(ind_high_start, ind_high_end, 0))
        ft = np.delete(ft[:], np.arange(ind_high_start, ind_high_end, 1))
        return ft


    def get_scaling_px2metr(self) -> float:
        """Calculates the frequency scaling from an input image/model and returns it in meters baseline per pixel"""
        return (self.fftscale/self.mas2rad)*self.wavelength

    def rescale_uvcoords(self, uvcoords: np.array) -> np.array:
        """Rescaled the uv-coords with the scaling factor and the max image size"""
        return uvcoords/(self.get_scaling_px2metr()*self.model_size)

    def distance(self, uvcoords: np.array) -> np.array:
        """Calculates the norm for a point. Takes the freq and checks both the u and v coords against it
        (works only for models/image that have the same lenght in both dimensions).

        The indizes of the output list evaluate to the indices of the input list"""
        # This makes a list of all the distances in the shape (size_img_array, uv_coords)
        freq_distance_lst = [[np.sqrt((j-i)**2) for j in self.fftfreq] for i in uvcoords]

        # This gets the indices of the elements closest to the uv-coords
        indices_lst = [[j for j, o in enumerate(i) if o == np.min(np.array(i))] for i in freq_distance_lst]
        return np.ndarray.flatten(np.array(indices_lst))

    def correspond_fft2freq(self, rescaled_uvcoords: np.array) -> list:
        """This calculates the closest point in the scaled, transformed uv-coords to the FFT result and returns the indicies of the FFT corresponding to the uv-coords"""
        u_ind = self.distance([i[0] for i in rescaled_uvcoords])
        v_ind = self.distance([i[1] for i in rescaled_uvcoords])
        return list(zip(u_ind, v_ind))

    def readout_all_px2uvcoords(self):
        """This function reads all the pixels into uv coords with the scaling factor"""
        ...

    @staticmethod
    def get_fft_values(ft: np.array, uv_ind: list) -> list:
        """Returns the FFT-values at the given indices"""
        return [ft[i[0]][i[1]] for i in uv_ind]

    @staticmethod
    def get_ft_amp_phase(ft: [np.array, int]) -> [np.array, np.array]:
        """Splits the real and imaginary part of the 2D-FFT into amplitude-,
         and phase-spectrum"""
        return np.abs(ft), np.angle(ft)


    def do_plot(self, ft: np.array, uvcoords: np.array) -> None:
        """Makes simple plots in the form of two subplots of the image before and after Fourier transformation"""
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.imshow(self.model)
        ax1.set_title(f"Model")
        ax1.set_xlabel(f"resolution [px] {self.model_size}")
        ax1.axes.get_xaxis().set_ticks([])
        ax1.axes.get_yaxis().set_ticks([])

        ax2.imshow(np.log(abs(ft)), interpolation='none', extent=[self.min_freq, self.max_freq, self.min_freq, self.max_freq])
        ax2.set_title("FFT")
        ax2.set_xlabel("freq")
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])

        x, y = np.array([i[0] for i in uvcoords]), np.array([i[1] for i in uvcoords])
        ax2.scatter(x, y, s=5)
        plt.show()


if __name__ == "__main__":
    # for i in range(134, 2011, 25):
    #     print("-----------------------------------------------------\n{}".format(i))
    #     fourier = FFT(modelling.UniformDisk(i, 150).eval_model(),"TARGET_CAL_INT_0001bcd_calibratedTEST.fits",  greyscale=False, step_size_fft=1)
    ring = Gauss2D()

    fourier = FFT(ring.eval_model(1024, 256.1), 1024,"assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits", 8e-06)

