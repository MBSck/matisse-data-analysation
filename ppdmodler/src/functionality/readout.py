#!/usr/bin/env python3

import numpy as np

from astropy.io import fits
from typing import Any, Dict, List, Union, Optional

from src.functionality.utilities import get_distance

class ReadoutFits:
    """All functionality to work with '.oifits/.fits'-files"""
    def __init__(self, fits_file):
        self.fits_file = fits_file

    def get_info(self):
        """Gets the header's info"""
        with fits.open(self.fits_file) as hdul:
            return hdul.info()

    def get_header(self, hdr):
        """Reads out the specified data"""
        return repr(fits.getheader(self.fits_file, hdr))

    def get_data(self, hdr: Union[int, str], *args: Union[int, str]) -> List[np.array]:
        """Gets a specific set of data and its error from a header and
        subheader and returns the data of as many subheaders as in args

        Parameters
        ----------
        hdr: int | str
            The header of the data to be retrieved
        args: int | str
            The subheader(s) that specify the data

        Returns
        -------
        data: List[np.array]
        """
        with fits.open(self.fits_file) as hdul:
            return [hdul[hdr].data[i] for i in args] if len(args) > 1 \
                    else hdul[hdr].data[args[0]]

    def get_column_names(self, hdr):
        """Fetches the columns of the header"""
        with fits.open(self.fits_file) as hdul:
            return (hdul[hdr].columns).names

    def get_uvcoords(self):
        """Fetches the u, v coord-lists and merges them as well as the individual components"""
        return np.array([i for i in zip(self.get_data(4, "ucoord")[:6], self.get_data(4, "vcoord")[:6])])

    def get_split_uvcoords(self):
        """Splits a 2D-np.array into its 1D-components and returns the u- and
        v-coords seperatly"""
        uvcoords = self.get_uvcoords()
        return np.array([item[0] for item in uvcoords]), np.array([item[1] for item in uvcoords])

    def get_vis(self):
        """Fetches the visibility data, its error and sta_index of the
        baselines"""
        ...

    def get_vis2(self):
        """Fetches the squared visibility data, its error and sta_index of the
        baselines"""
        return self.get_data("oi_vis2", "vis2data", "vis2err", "sta_index")

    def get_t3phi(self):
        """Fetches the closure phase data, its error and sta_index of the
        phases"""
        return self.get_data("oi_t3", "t3phi", "t3phierr", "sta_index")

    def get_wl(self):
        return self.get_data("oi_wavelength", "eff_wave")

    def get_tel_sta(self):
        return self.get_data(2, "tel_name", "sta_index")

    def get_vis24wl(self, wl_ind: int) -> np.ndarray:
        """Fetches the vis2data for one specific wavelength

        Returns
        --------
        np.ndarray
            Array with two nested arrays containing the the vis2  and the
            vis2err for all baselines for a specific wavelength
        """
        vis2data, vis2err  = map(lambda x: x[:6], self.get_vis2()[:2])
        vis2datawl, vis2errwl = map(lambda x: np.array([i[wl_ind] for i in x]).flatten(), [vis2data, vis2err])

        return vis2datawl, vis2errwl


if __name__ == "__main__":
    file = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_RAW_INT_0001.fits"
    readout = ReadoutFits(file)
    print(readout.get_uvcoords())
    # readout.get_info()

    print(readout.get_vis24wl())
