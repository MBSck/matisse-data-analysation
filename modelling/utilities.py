#!/usr/bin/env python3

__author__ = "Marten Scheuck"

import numpy as np
from astropy.io import fits


class ReadoutFits:
    def __init__(self, fits_file):
        self.fits_file = fits_file

    def read_fits_info(self):
        with fits.open(self.fits_file) as hdul:
            return hdul.info()

    def read_fits_header(self, header):
        """Reads out the specified data from the '.fits'-file"""
        with fits.open(self.fits_file) as hdul:
            return hdul[header].data


if __name__ == "__main__":
    readout = ReadoutFits("TARGET_CAL_INT_0001bcd_calibratedTEST.fits")
    print(readout.read_fits_header("oi_vis2"))
    print(readout.read_fits_info())
