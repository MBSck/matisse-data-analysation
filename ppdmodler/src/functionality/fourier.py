#!/usr/bin/env python3

# Credit
# FFT: https://numpy.org/doc/stable/reference/routines.fft.html

import numpy as np                  # Imports the numpy module
import time                         # Imports the time module
import matplotlib.pyplot as plt     # Imports the matplotlib module

from numpy import fft               # Imports the Fast-Fourier Transform (FFT) pack
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

# Own modules
from src.functionality.utilities import ReadoutFits, timeit, do_plot
from src.models.gauss2d import Gauss2D   # For testing the functionality of fourier.py

# Be aware of the fact that the code only works for same dimensional pictures/models

# TODO: Make class and function documentation
# TODO: Make the class return the rescaled information readily, maybe rescale
# it outside of the FFT class?

class FFT:
    """A collection of the fft-functionality given by scipy

    Attributes
    ----------

    """
    # TODO: Check how the indices are sorted, why do they change? They change even independent of the scaling
    # TODO: Remove staticmethods and make them outside of class?
    # TODO: Change euclidean distance to interpolation in order to get coordinates
    # TODO: Make Zoom function work
    def __init__(self, model: np.array, set_size: int, fits_file_path: Path,
                 wavelength: float, step_size_fft: float = 1., greyscale: bool = False) -> None:
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

        self.wavelength = wavelength                                                # The set wavelength for scaling
        self.mas2rad = np.deg2rad(1/3.6e6) # mas per rad

        # Sets the variables that take computational power
        self.ft = self.do_fft2()
        self.scaling = self.get_scaling_px2metr()
        self.uvcoords = self.readout.get_uvcoords()                                 # Gets the uvcoords of vis2
        self.rescaled_uvcoords = self.rescale_uvcoords

    def fft_pipeline(self) -> [float, np.array, float, float]:
        """A pipeline function that calls the functions of the FFT in order, and
        avoids double calling of single functions"""
        uv_ind = self.correspond_fft2freq(rescaled_uvcoords)
        fft_value = FFT.get_fft_values(self.ft, uv_ind)
        # ft = self.zoom_fft2(self.ft, self.set_size)
        return self.ft, self.rescaled_uvcoords, fft_value

    @timeit
    def do_fft2(self) -> np.array:
        """Does the 2D-FFT and returns the 2D-FFT"""
        return fft.fftshift(fft.fft2(fft.ifftshift(self.model)))

    @timeit
    def do_ifft2(self) -> np.array:
        """Does the inverse 2D-FFT and returns the inverse 2D-FFT"""
        return fft.fftshift(fft.ifft2(fft.fftshift(self.model))).real

    def zoom_fft2(ft: np.array, self, set_size: int):
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

    def rescale_uvcoords(self) -> np.array:
        """Rescaled the uv-coords with the scaling factor and the max image size"""
        return self.uvcoords/(self.scaling*self.model_size)

    def distance(self) -> np.array:
        """Calculates the norm for a point. Takes the freq and checks both the
        u- and v-coords against it (works only for models/image that have the same length in both dimensions).

        The indices of the output list evaluate to the indices of the input list"""
        # This makes a list of all the distances in the shape (size_img_array, uv_coords)
        freq_distance_lst = [[np.sqrt((j-i)**2) for j in self.fftfreq] for i in
                             self.uvcoords]

        # This gets the indices of the elements closest to the uv-coords
        indices_lst = [[j for j, o in enumerate(i) if o == np.min(np.array(i))] for i in freq_distance_lst]
        return np.ndarray.flatten(np.array(indices_lst))

    def correspond_fft2freq(self) -> List:
        """This calculates the closest point in the scaled, transformed
        uv-coords to the FFT result and returns the indicies of the FFT corresponding to the uv-coords"""
        u_ind = self.distance([i[0] for i in rescaled_uvcoords])
        v_ind = self.distance([i[1] for i in rescaled_uvcoords])
        return list(zip(u_ind, v_ind))

    def readout_all_px2uvcoords(self):
        """This function reads all the pixels into uv coords with the scaling factor"""
        ...

    @staticmethod
    def get_fft_values(ft: np.array, uv_ind: List) -> List:
        """Returns the FFT-values at the given indices"""
        return [ft[i[0]][i[1]] for i in uv_ind]

    @staticmethod
    def get_ft_amp_phase(ft: [np.array, int]) -> [np.array, np.array]:
        """Splits the real and imaginary part of the 2D-FFT into amplitude-,
         and phase-spectrum"""
        return np.abs(ft), np.angle(ft)


if __name__ == "__main__":
    ring = Gauss2D()
    file_path = "/Users/scheuck/Documents/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    fourier = FFT(ring.eval_model(1024, 256.1), 1024, file_path, 8e-06)
