#!/usr/bin/env python3

__author__ = "Marten Scheuck"

import numpy as np                  # Imports the numpy module
import matplotlib.pyplot as plt     # Imports the matplotlib module
from scipy import fft               # Imports the Fast-Fourier Transform (FFT) pack from the scipy module

import modelling


def read_image_into_nparray(image):
    """This checks the input if it is an np.array and if not reads it in as such"""
    if isinstance(image, np.ndarray):
        return image

    return plt.imread(image)


def greyscale_image_array(image_array):
    """Turns an image or an array of it into greyscale"""
    return read_image_into_nparray(image_array)


def do_fft2(image, greyscale=True):
    """Does the 2D-FFT and returns the input image in greyscale and the 2D-FFT"""
    if greyscale:
        image = read_image_into_nparray(image)[:, :, :3].mean(axis=2)  # Simple grayscale conversion

    plt.set_cmap("gray")

    return image, fft.fftshift(fft.fft2(fft.ifftshift(image)))


def do_ifft2(image, greyscale=True):
    """Does the inverse 2D-FFT and returns the input image and the inverse 2D-FFT"""
    if greyscale:
        image = read_image_into_nparray(image)[:, :, :3].mean(axis=2)   # Simple greyscale conversion

    plt.set_cmap("gray")

    return image, fft.fftshift(fft.ifft2(fft.fftshift(image))).real


def do_plot(image, fourier_image):
    """This plots two subplots of the image before and after Fourier-transformation"""
    plt.subplot(121)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(122)
    # plt.imshow(abs(fourier_image))
    plt.imshow(np.log(abs(fourier_image)))  # np.log is used when max amplitude would be too bright
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    img, ft = do_fft2("Michelson.png")
    # img, ft = do_fft2(modelling.model_generation(modelling.gauss2d), greyscale=False)
    do_plot(img, ft)
