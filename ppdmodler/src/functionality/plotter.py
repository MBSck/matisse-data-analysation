#!/usr/bin/env python3

# TODO: Make either model or fourier transform carry more info like the name of
# the plot or similar -> Work more with classes

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from glob import glob
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.special import j0, j1
from skimage.restoration import unwrap_phase

from src.functionality.readout import ReadoutFits

def shell_main() -> None:
    """
    This function's sole purpose is to enable the plotter to work in the shell
    """
    try:
        script, dirname, vis_dim0, vis_dim1 = sys.argv
    except:
        print("Usage: python3 plotter.py /path/to/target/data/dir/ vis_dim[0] vis_dim[1]")
        sys.exit(1)

    vis_dim = [float(vis_dim0), float(vis_dim1)]
    Plotter(dirname=dirname, vis_dim=vis_dim)

def gaussian(spat_freq: np.array, D: float) -> np.array:
    """ A gaussian fit described by the 0th-order Bessel function

    Parameters
    ----------
    spat_freq: np.array
        Spatial frequency
    D: float
        Distance

    Returns
    -------
    gaussian_fit: np.array
    """
    return np.exp(-np.square(np.pi*D*spat_freq)/(4* np.log(2)))

def airy(spat_freq: np.array, D: float) -> np.array:
    """An airy disk fit described by the 1st-order Bessel function

    Parameters
    ----------
    spat_freq: np.array
        Spatial frequency
    D: float
        Distance

    Returns
    -------
    airy_disk_fit: np.array
    """
    radial_dist = spat_freq*D
    return  2*j1(np.pi*radial_dist)/radial_dist/np.pi

# Classes

class Plotter:
    """Class that plots models as well as vis-, t3phi- and uv-data"""
    def __init__(self, dirname: Path, vis_dim: List[float]) -> None:
        self.files = np.sort(glob(dirname + "/*CAL_INT*.fits"))
        self.dirname = dirname
        self.vis_dim = vis_dim

        if self.files is None:
            print("No files found! Check input path")
            sys.exit(1)

        for file in self.files[:]:
            print(f'Reading files from "{self.dirname}"')
            self.fits_file = file

            # Initializes the fits-readout
            self.readout = ReadoutFits(file)

            # Fetches all the relevant data from the '.fits'-file
            self.vis2data, self.vis2err = map(lambda x: x[:6], self.readout.get_vis2()[:2])
            self.t3phidata, self.t3phierr = map(lambda x: x[:4], self.readout.get_t3phi()[:2])
            self.vis2sta, self.t3phista = self.readout.get_vis2()[2], self.readout.get_t3phi()[2]
            self.ucoords, self.vcoords = map(lambda x: x[:6],
                                             self.readout.get_split_uvcoords())
            self.wl = self.readout.get_wl()

            # Different baseline-configurations (small-, medium-, large) AT & UT. Telescope names and "sta_index"
            self.all_tels = {}
            smallAT, medAT, largeAT, UT = {1: "A0", 5: "B2", 13: "C0", 10: "D1"}, \
                    {28: "K0", 18: "G1", 13: "D0", 24: "J3"}, \
                    {1: "A0", 18: "G1", 23: "J2", 24: "J3"}, \
                    {32: "UT1", 33: "UT2", 34: "UT3", 35: "UT4"}
            self.all_tels.update(smallAT)
            self.all_tels.update(medAT)
            self.all_tels.update(largeAT)
            self.all_tels.update(UT)

            '''
            # Code for python 3.9, dic merging operator
            self.all_tels = {1: "A0", 5: "B2", 13: "C0", 10: "D1"} | \
                    {28: "K0", 18: "G1", 13: "D0", 24: "J3"} | \
                    {1: "A0", 18: "G1", 23: "J2", 24: "J3"} | \
                    {32: "UT1", 33: "UT2", 34: "UT3", 35: "UT4"}
            '''

            # Sets the descriptors of the telescopes' baselines and the closure # phases
            self.tel_vis2 = np.array([("-".join([self.all_tels[t] for t in duo])) for duo in self.vis2sta])
            self.tel_t3phi = np.array([("-".join([self.all_tels[t] for t in trio])) for trio in self.t3phista])

            # The mean of the wavelength. The mean of all the visibilities and their standard deviation
            self.wl_slice= [j for j in self.wl if (j >= np.mean(self.wl)-0.5e-06 and j <= np.mean(self.wl)+0.5e-06)]
            self.si, self.ei = (int(np.where(self.wl == self.wl_slice[0])[0])-5,
                                    int(np.where(self.wl == self.wl_slice[~0])[0])+5)

            self.mean_bin_vis2 = [np.nanmean(i[self.si:self.ei]) for i in self.vis2data]
            self.std_bin_vis2 = [np.nanmean(j[self.si:self.ei]) for j in self.vis2data]
            self.baseline_distances = [np.sqrt(x**2+y**2) for x, y in zip(self.ucoords, self.vcoords)]

            # Executes the plotting and cleans everything up
            self.do_plot()
            self.close()

    def do_plot(self):
        """This is the main pipline of the class. It brings all the plots together into one consistent one"""
        print(f"Plotting {os.path.basename(Path(self.fits_file))}")

        fig, axarr = plt.subplots(3, 6, figsize=(20, 10))

        # Flattens the multidimensional arrays into 1D
        ax, bx, cx, dx, ex, fx, = axarr[0].flatten()
        ax2, bx2, cx2, dx2, ex2, fx2 = axarr[1].flatten()
        ax3, bx3, cx3, dx3, ex3, fx3 = axarr[2].flatten()

        self.vis2_plot(axarr)
        self.t3phi_plot(axarr)
        self.fits_plot(ex2)
        self.waterfall_plot(fx2)
        self.uv_plot(ax3)
        # self.model_plot(bx3, cx3, gauss)
        # self.model_plot(dx3, ex3, ring)
        print(f"Done plotting {os.path.basename(Path(self.fits_file))}")

    def vis2_plot(self, axarr) -> None:
        """Plots the squared visibilities"""
        # Sets the range for the squared visibility plots
        all_obs = [[],[],[],[],[],[]]

        # plots the squared visibility for different degrees and meters
        for i, o in enumerate(self.vis2data):
            axis = axarr[0, i%6]
            axis.errorbar(self.wl*1e6, o, yerr=self.vis2err[i], marker='s',
                          label=self.tel_vis2[i], capsize=0., alpha=0.5)
            axis.set_ylim([self.vis_dim[0], self.vis_dim[1]])
            axis.set_ylabel("vis2")
            axis.set_xlabel("wl [micron]")
            all_obs[i%6].append(o)

        # Plots the squared visibility for different degrees and metres
        for j in range(6):
            axis = axarr[0, j%6]
            pas = (np.degrees(np.arctan2(self.vcoords[j], self.ucoords[j]))-90)*-1
            axis.errorbar(self.wl*1e6, np.nanmean(all_obs[j], 0), yerr=np.nanstd(all_obs[j], 0),
                          marker='s', capsize=0., alpha=0.9, color='k',
                          label='%.1f m %.1f deg'%(np.sqrt(self.ucoords[j]**2+self.vcoords[j]**2), pas))
            axis.legend(loc=2)

    def t3phi_plot(self, axarr) -> None:
        """Plots the closure phase"""
        all_obs = [[],[],[],[]]
        for i, o in enumerate(self.t3phidata):
            axis = axarr[1, i%4]
            axis.errorbar(self.wl*1e6, unwrap_phase(o), yerr=self.t3phierr[i],marker='s',capsize=0.,alpha=0.25)
            axis.set_ylim([-180,180])
            axis.set_ylabel("cphase [deg]")
            axis.set_xlabel("wl [micron]")
            all_obs[i%4].append(list(o))

        for j in range(4):
            axis = axarr[1, j%4]
            axis.errorbar(self.wl*1e6, all_obs[j][0], yerr=np.std(all_obs[j], 0),
                          marker='s', capsize=0., alpha=0.9, color='k',
                          label=self.tel_t3phi[j])
            axis.legend([self.tel_t3phi[j]], loc=2)

    def waterfall_plot(self, ax) -> None:
        # Plot waterfall with the mean wavelength for the different baselines

        for i in range(6):
            ax.errorbar(self.wl[self.si:self.ei]*1e06, self.vis2data[i][self.si:self.ei],
                         yerr=np.nanstd(self.vis2data[i][self.si:self.ei]),
                         label=self.tel_vis2[i], ls='None', fmt='o')
            ax.set_xlabel(r'wl [micron]')
            ax.set_ylabel('vis2')
            ax.legend(loc='best')


    def fits_plot(self, ax) -> None:
        # Plot the mean visibility for one certain wavelength and fit it with a gaussian and airy disk
        ax.errorbar(self.baseline_distances, self.mean_bin_vis2, yerr=self.std_bin_vis2, ls='None', fmt='o')

        # Fits the data
        scaling_rad2arc = 206265

        # Gaussian fit
        fwhm = 1/scaling_rad2arc/1000           # radians

        # np.linspace(np.min(spat_freq), np.max(spat_freq), 25)
        xvals = np.linspace(50, 3*150)/3.6e-6
        fitted_model= np.square(gaussian(xvals, fwhm))
        ax.plot(xvals/1e6, fitted_model*0.15, label='Gaussian %.1f"'%(fwhm*scaling_rad2arc*1000))

        # Airy-disk fit
        fwhm = 3/scaling_rad2arc/1000           # radians
        fitted_model = np.square(airy(xvals, fwhm))
        ax.plot(xvals/1e6, fitted_model*0.15, label='Airy Disk %.1f"'%(fwhm*scaling_rad2arc*1000))
        ax.set_ylim([0, 0.175])
        ax.legend(loc='best')

        ax.set_xlabel(fr'uv-distance [m] at $\lambda_0$={10.72} $\mu m$')
        ax.set_ylabel(r'$\bar{V}$')

    def uv_plot(self, ax) -> None:
        """Plots the uv-coordinates with an orientational compass

        Parameters
        ----------
        ax
            The axis anchor of matplotlib.pyplot

        Returns
        -------
        None
        """
        ax.scatter(self.ucoords, self.vcoords)
        ax.scatter(-self.ucoords, -self.vcoords)
        ax.set_xlim([150, -150])
        ax.set_ylim([-150, 150])
        ax.set_ylabel('v [m]')
        ax.set_xlabel('u [m]')

        # Compasss for the directions
        cardinal_vectors = [(0,1), (0,-1), (1,0), (-1,0)]
        cardinal_colors  = ['black', 'green', 'blue', 'red']
        cardinal_directions = ['N', 'S', 'W', 'E']
        arrow_size, head_size = 40, 10
        x, y = (-85, 85)

        for vector, color, direction in zip(cardinal_vectors, cardinal_colors, cardinal_directions):
            dx, dy = vector[0]*arrow_size, vector[1]*arrow_size
            if vector[0] == 0:
                ax.text(x-dx-5, y+dy, direction)
            if vector[1] == 0:
                ax.text(x-dx, y+dy+5, direction)
            arrow_args = {"length_includes_head": True, "head_width": head_size, "head_length": head_size, \
                                  "width": 1, "fc": color, "ec": color}
            ax.arrow(x, y, dx, dy, **arrow_args)

    def model_plot(self, ax1, ax2, model) -> None:
        """"""
        # Plots the model fits and their fft of the uv-coords
        # TODO: Implement this rescaling somehow differently so that there
        # needs to be no import of main
        img, ft, rescaled_uvcoords = main(f, "ring")

        # Plots the gaussian model
        ax1.imshow()
        ax1.set_title(f'')
        ax1.set_xlabel(f"resolution [px] 1024, zero padding 2048")
        ax1.axes.get_xaxis().set_ticks([])
        ax1.axes.get_yaxis().set_ticks([])

        ax2.imshow(np.log(abs(gauss_ft)), interpolation='none', extent=[-0.5, 0.5, -0.5, 0.5])
        # Rename the plots with the future information -> in all cases
        ax2.set_title("Gauss FFT")
        ax2.set_xlabel("freq")
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])

        u, v = np.array([i[0] for i in uvcoords]), np.array([i[1] for i in uvcoords])
        ax2.scatter(u, v, s=5)

    def write_values(self) -> None:
        """"""
        with open(outname[:~7]+"_phase_values.txt", 'w') as f:
            for i in range(4):
                f.write(str(unwrap_phase(t3phi[i])) + '\n')

    def close(self) -> None:
        """Finishing up the plot and then saving it to the designated folder"""
        plt.tight_layout()
        outname = self.dirname+'/'+self.fits_file.split('/')[-1]+'_qa.png'

        plt.savefig(outname, bbox_inches='tight')
        plt.close()


if __name__ == ('__main__'):
    ...
    # Tests
    # ------
    data_path = "data/beegfs/astro-storage/groups/matisse/scheuck/data/openTime/suAur/PRODUCTS/nband/calib_nband/"
    # folders = [os.path.join(data_path, "2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T04_52_11.AQUARIUS.rb_CALIBRATED"),
    #           os.path.join(data_path, "2019-05-14T04_52_11.AQUARIUS.rb_with_2019-05-14T06_12_59.AQUARIUS.rb_CALIBRATED"),
    #           os.path.join(data_path, "2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T06_12_59.AQUARIUS.rb_CALIBRATED")]

    # for i in folders:
    #     Plotter(i, [0., 0.15])

    folder = "2021-10-15T07_20_19.AQUARIUS.rb_with_2021-10-15T06_50_56.AQUARIUS.rb_CALIBRATED"
    Plotter(os.path.join(data_path, folder), [0., 0.15])
    # ------

    # Main process for shell usage
    # shell_main()

