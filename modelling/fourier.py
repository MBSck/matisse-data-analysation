#!/usr/bin/env python3

__author__ = "Marten Scheuck"

import numpy as np                  # Imports the numpy module
import matplotlib.pyplot as plt     # Imports the matplotlib module
from numpy import fft               # Imports the Fast-Fourier Transform (FFT) pack

import modelling
from utilities import ImageProcessing, ReadoutFits

class FFT:
    """A collection of the fft-functionality given by scipy"""
    # TODO: Check how the 'step_size_fft' changes the scaling
    # TODO: Be aware of the fact that the code only works for same dimensional pictures/models
    def __init__(self, path_to_image, fits_file_path, greyscale=True, step_size_fft: float = 1.0):
        # Checks if the input is an image or a model, i.e. an array
        try:
            self.img_proc = ImageProcessing(path_to_image)
            self.img_size = self.img_proc.get_img_size[0]                     # Takes the first of both axis as the images size
        except:
            self.img_size = len(path_to_image)

        self.readout = ReadoutFits(fits_file_path)

        if greyscale:
            self.img_array = self.img_proc.read_image_into_nparray()[:, :, :3].mean(axis=2)    # Converts the image to a numpy array and to greyscale
        else:
            self.img_array = self.img_proc.read_image_into_nparray()                           # Checks if the input is a numpy.array and if not converts it to one
        plt.set_cmap("gray")
        
        # General variables
        self.setp_size_fft = step_size_fft
        self.fftfreq = fft.fftshift(fft.fftfreq(self.img_size, d=step_size_fft))               # Set the x-axis of the FFT, corresponds to the x-axis before the FFT

        # Conversion units
        self.mas2rad = np.deg2rad(1/3600000) # mas per rad


    def do_fft2(self):
        """Does the 2D-FFT and returns the 2D-FFT"""
        return fft.fftshift(fft.fft2(fft.ifftshift(self.img_array)))

    def do_ifft2(self):
        """Does the inverse 2D-FFT and returns the inverse 2D-FFT"""
        return fft.fftshift(fft.ifft2(fft.fftshift(self.img_array))).real
    
    def get_px_scaling_to_meter(self, wavelength: float = 8**10**(-6)):
        """Calculates the frequency scaling from an input image/model and returns it in meters baseline per pixel""" 
        roll = np.floor(self.img_size/2).astype(int)                            # Gets the max width of the picture and takes the half of it to get the distance from the center, then puts it out as an int 
        freq = np.roll(self.fftfreq, roll, axis=0)                              # ?
        fftscale = np.diff(freq)[0]                                             # cycles / mas per pixel in FFT image, takes the diff between the adjacent elements (are the same for all cases)
        return fftscale/self.mas2rad * wavelength                               # meters baseline per pixle in FFT image at given wavelength
        

    def rescale_uv_coords(self, wavelength: float = 8**10**(-6)):
        """Rescaled the uv-coords with the scaling factor"""
        return self.readout.convert_uvcoords_vis2_to_rads()/self.get_px_scaling_to_meter() 

    def fft2_to_amplitude_phase(self, ft):
        """Splits the real and imaginary part of the 2D-FFT into amplitude-,
        power- and phase-spectrum"""
        return np.abs(ft), np.abs(ft)**2, np.angle(ft)

    def correspond_fft_to_freq(self):
        """This calculates the closest point in the scaled, transformed uv-coords to the FFT result and returns the indicies of the FFT corresponding to the uv-coords"""
        fftaxes = np.array([i for i in zip(self.fftfreq, self.fftfreq)])       # Works for images/models that are the same in both dimension
        # return np.linalg.norm()
        return np.linalg.norm(self.rescale_uv_coords()[0], fftaxes)
        
    def do_plot(self, fourier_img: np.array, uvcoords: np.array):
        """Makes simple plots in the form of two subplots of the image before and after Fourier transformation"""
        # Plots the img before the FFT
        # plt.subplot(121)
        # plt.imshow(self.img)
        # plt.axis("off")

        # Plots the img after the FFT
        # plt.subplot(122)
        plt.imshow(abs(fourier_img), interpolation='none', extent=[-0.5, 0.5, -0.5, 0.5])
        # plt.imshow(np.log(abs(fourier_img_array)))  # np.log is used when max. amplitude would be too bright
        # plt.axis("off")
        # plt.plot(image_array_1D, image_array_1D)

        # Plots the uv-coords as a scatter plot onto the after img
        self.readout.do_uv_plot(uvcoords)


if __name__ == "__main__":
    # Fourier test
    # fourier = FFT("Michelson.png", "TARGET_CAL_INT_0001bcd_calibratedTEST.fits")
    # ft = fourier.do_fft2()
    fourier2 = FFT(modelling.model_generation(modelling.gauss2d),"TARGET_CAL_INT_0001bcd_calibratedTEST.fits",  greyscale=False, step_size_fft=1.0)
    ft2 = fourier2.do_fft2()

    # print(fourier2.img_size)

    # Functionality test
    # print(fourier.fft2_to_amplitude_phase(ft))
    # print(readout.convert_uvcoords_vis2_to_rads())
    print(fourier2.readout.convert_uvcoords_vis2_to_rads()/fourier2.get_px_scaling_to_meter(), "Converted Coords divided by scaling")
    # print(fourier.readout.convert_uvcoords_vis2_to_rads(), "Converted coordinates")
    # print(fourier.get_px_scaling_to_meter(), "FFT Scaling in meters per baseline") 
    print(fourier2.fftfreq, "Frequency x-axis")
    # fourier2.do_plot(ft2, fourier2.rescale_uv_coords())
    print(fourier2.correspond_fft_to_freq())
