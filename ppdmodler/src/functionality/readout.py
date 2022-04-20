#!/usr/bin/env python3

import numpy as np

from scipy.interpolate import CubicSpline
from astropy.io import fits
from typing import Any, Dict, List, Union, Optional

from src.functionality.utilities import get_distance

def read_single_dish_txt2np(file, wl_axis):
    """Reads x, y '.txt'-file intwo 2 numpy arrays"""
    file_data = np.loadtxt(file)
    wavelength_axis = np.array([i[0] for i in file_data])*1e-6
    flux_axis = np.array([i[1] for i in file_data])

    wl2flux_dict = {}
    cs = CubicSpline(wavelength_axis, flux_axis)
    for i, o in enumerate(cs(wl_axis)):
        wl2flux_dict[wl_axis[i]] = o

    return wl2flux_dict

class ReadoutFits:
    """All functionality to work with '.oifits/.fits'-files"""
    def __init__(self, fits_file) -> None:
        self.fits_file = fits_file

    def get_info(self) -> str:
        """Gets the header's info"""
        with fits.open(self.fits_file) as hdul:
            return hdul.info()

    def get_header(self, hdr) -> str:
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

    def get_column_names(self, hdr) -> np.ndarray:
        """Fetches the columns of the header"""
        with fits.open(self.fits_file) as hdul:
            return (hdul[hdr].columns).names

    def get_uvcoords(self) -> np.ndarray:
        """Fetches the u, v coord-lists and merges them as well as the individual components"""
        return np.array([i for i in zip(self.get_data(4, "ucoord")[:6], self.get_data(4, "vcoord")[:6])])

    def get_split_uvcoords(self) -> np.ndarray:
        """Splits a 2D-np.array into its 1D-components and returns the u- and
        v-coords seperatly"""
        uvcoords = self.get_uvcoords()
        return np.array([item[0] for item in uvcoords]), np.array([item[1] for item in uvcoords])

    def get_vis(self) -> np.ndarray:
        """Fetches the visibility data/correlated fluxes, its errors and sta-indices"""
        return self.get_data("oi_vis", "visamp", "visamperr", "visphi", "visphierr", "sta_index")

    def get_vis2(self) -> np.ndarray:
        """Fetches the squared visibility data, its error and sta_indicies"""
        return self.get_data("oi_vis2", "vis2data", "vis2err", "sta_index")

    def get_t3phi(self) -> np.ndarray:
        """Fetches the closure phase data, its error and sta_indicies"""
        return self.get_data("oi_t3", "t3phi", "t3phierr", "sta_index")

    def get_flux(self) -> np.ndarray:
        """Fetches the flux"""
        return self.get_data("oi_flux")

    def get_wl(self) -> np.ndarray:
        return self.get_data("oi_wavelength", "eff_wave")

    def get_tel_sta(self) -> np.ndarray:
        return self.get_data(2, "tel_name", "sta_index")

    def get_flux4wl(self, wl_ind: int) -> np.ndarray:
        """Fetches the flux for a specific wavelength"""
        return self.get_flux[wl_ind]

    def get_vis4wl(self, wl_ind: int) -> np.ndarray:
        """Fetches the visdata(amp/phase)/correlated fluxes for one specific wavelength

        Returns
        --------
        visamp4wl: np.ndarray
            The visamp for a specific wavelength
        visamperr4wl: np.ndarray
            The visamperr for a specific wavelength
        visphase4wl: np.ndarray
            The visphase for a specific wavelength
        visphaseerr4wl: np.ndarray
            The visphaseerr for a specific wavelength
        """
        visdata = self.get_vis()
        visamp, visamperr = map(lambda x: x[:6], visdata[:2])
        visphase, visphaseerr = map(lambda x: x[:6], visdata[2:4])
        visamp4wl, visamperr4wl = map(lambda x: np.array([i[wl_ind] for i in x]).flatten(), [visamp, visamperr])
        visphase4wl, visphaseerr4wl= map(lambda x: np.array([i[wl_ind] for i in x]).flatten(), [visphase, visphaseerr])

        return visamp4wl, visamperr4wl, visphase4wl, visphaseerr4wl

    def get_vis24wl(self, wl_ind: int) -> np.ndarray:
        """Fetches the vis2data for one specific wavelength

        Returns
        --------
        vis2data4wl: np.ndarray
            The vis2data for a specific wavelength
        vis2err4wl: np.ndarray
            The vis2err for a specific wavelength
        """
        vis2data, vis2err  = map(lambda x: x[:6], self.get_vis2()[:2])
        vis2data4wl, vis2err4wl = map(lambda x: np.array([i[wl_ind] for i in x]).flatten(), [vis2data, vis2err])

        return vis2data4wl, vis2err4wl


if __name__ == "__main__":
    readout = ReadoutFits("/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T06_12_59.rb/averaged/Final_CAL.fits")
    print(readout.get_vis()[1][5][110])
    print(readout.get_vis4wl(110))

