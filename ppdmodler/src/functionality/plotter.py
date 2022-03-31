#!/usr/bin/env python3

# TODO: Make either model or fourier transform carry more info like the name of
# the plot or similar -> Work more with classes
# TODO: Remove error bars from plots
# TODO: Make plot save mechanic with automatic generated names specific to input file
# TODO: Rework dump function into pickle dump

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
from src.functionality.utilities import trunc

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

# Classes

class Plotter:
    """Class that plots models as well as vis-, t3phi- and uv-data"""
    def __init__(self, dirname: Path, mosaic: bool = False) -> None:
        self.files = np.sort(glob(dirname + "/*CAL_INT*.fits"))
        self.dirname = dirname
        self.mosaic = mosaic
        self.outname = ""
        print(f'Reading files from "{self.dirname}"')

        if self.files is None:
            print("No files found! Check input path")
            sys.exit(1)

        for f in self.files[:]:
            self.fits_file = f

            # Initializes the fits-readout
            self.readout = ReadoutFits(f)

            # NOTE: Debug only remove later
            self.vis = self.readout.get_data("oi_vis", "visamp", "visamperr", "visphi", "visphierr", "sta_index")

            # Fetches all the relevant data from the '.fits'-file
            self.vis2data, self.vis2err = map(lambda x: x[:6], self.readout.get_vis2()[:2])
            self.visdata, self.viserr = map(lambda x: x[:6], self.vis[:2])
            self.visphase, self.visphaseerr = map(lambda x: x[:6], self.vis[2:4])
            self.t3phidata, self.t3phierr = map(lambda x: x[:4], self.readout.get_t3phi()[:2])
            self.vis2sta, self.t3phista = self.readout.get_vis2()[2], self.readout.get_t3phi()[2]
            self.vissta = self.vis[~0]
            self.ucoords, self.vcoords = map(lambda x: x[:6], self.readout.get_split_uvcoords())
            self.wl = self.readout.get_wl()[11:-17]

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
            # Code for python 3.9 and higher, dic merging operator
            self.all_tels = {1: "A0", 5: "B2", 13: "C0", 10: "D1"} | \
                    {28: "K0", 18: "G1", 13: "D0", 24: "J3"} | \
                    {1: "A0", 18: "G1", 23: "J2", 24: "J3"} | \
                    {32: "UT1", 33: "UT2", 34: "UT3", 35: "UT4"}
            '''

            # Sets the descriptors of the telescopes' baselines and the closure # phases
            self.tel_vis = np.array([("-".join([self.all_tels[t] for t in duo])) for duo in self.vissta])
            self.tel_vis2 = np.array([("-".join([self.all_tels[t] for t in duo])) for duo in self.vis2sta])
            self.tel_t3phi = np.array([("-".join([self.all_tels[t] for t in trio])) for trio in self.t3phista])

            # The mean of the wavelength. The mean of all the visibilities and their standard deviation
            self.mean_wl = np.mean(self.wl)
            self.wl_slice= [j for j in self.wl if (j >= self.mean_wl-0.5e-06 and j <= self.mean_wl+0.5e-06)]
            self.si, self.ei = (int(np.where(self.wl == self.wl_slice[0])[0])-5,
                                    int(np.where(self.wl == self.wl_slice[~0])[0])+5)

            self.mean_bin_vis2 = [np.nanmean(i[self.si:self.ei]) for i in self.vis2data]
            self.baseline_distances = [np.sqrt(x**2+y**2) for x, y in zip(self.ucoords, self.vcoords)]

            # Executes the plotting and cleans everything up
            if self.mosaic:
                self.do_mosaic_plot()
            else:
                self.do_plot()
            self.close()

    def do_plot(self) -> None:
        """This is the main pipline of the class. It brings all the plots together into one consistent one"""
        print(f"Plotting {os.path.basename(Path(self.fits_file))}")

        fig, axarr = plt.subplots(2, 6, figsize=(20, 8))
        ax, bx, cx, dx, ex, fx, = axarr[0].flatten()
        ax2, bx2, cx2, dx2, ex2, fx2 = axarr[1].flatten()

        self.vis2_plot(axarr)
        self.t3phi_plot(axarr)
        self.vis24baseline_plot(ex2)
        self.uv_plot(fx2)
        print(f"Done plotting {os.path.basename(Path(self.fits_file))}")

    def do_mosaic_plot(self) -> None:
        """Does a mosaic plot"""
        print(f"Plotting {os.path.basename(Path(self.fits_file))}")

        fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
        ax, bx, cx = axarr[0].flatten()
        ax2, bx2, cx2 = axarr[1].flatten()

        self.uv_plot(cx)
        self.vis_plot_all(bx)
        self.vis2_plot_all(ax2)
        self.t3phi_plot_all(bx2)
        self.vis24baseline_plot(cx2)
        print(f"Done plotting {os.path.basename(Path(self.fits_file))}")

    def vis_plot_all(self, ax) -> None:
        plot_dim = [np.min(self.visdata), np.max(self.visdata)]

        for i, o in enumerate(self.visdata):
            baseline = np.around(np.sqrt(self.ucoords[i]**2+self.vcoords[i]**2), 2)
            pas = np.around((np.degrees(np.arctan2(self.vcoords[i], self.ucoords[i]))-90)*-1, 2)
            ax.plot(self.wl*1e6, o[11:-17], label=fr"{self.tel_vis[i]}, $B_p$={baseline} m $\phi={pas}^\circ$",
                    linewidth=2)
            ax.set_ylim([*plot_dim])
            ax.set_ylabel("corr. flux [Jy]")
            ax.set_xlabel("wl [$\mu$ m]")

        ax.legend(loc=2, prop={'size': 6})

    def vis2_plot_all(self, ax) -> None:
        plot_dim = [np.min(self.vis2data), np.max(self.vis2data)]

        for i, o in enumerate(self.vis2data):
            baseline = np.around(np.sqrt(self.ucoords[i]**2+self.vcoords[i]**2), 2)
            pas = np.around((np.degrees(np.arctan2(self.vcoords[i], self.ucoords[i]))-90)*-1, 2)
            ax.plot(self.wl*1e6, o[11:-17], label=fr"{self.tel_vis2[i]}, $B_p$={baseline} m $\phi={pas}^\circ$",
                    linewidth=2)
            ax.set_ylim([*plot_dim])
            ax.set_ylabel("vis2")
            ax.set_xlabel(r"wl [$\mu$ m]")

        ax.legend(loc=2, prop={'size': 6})

    def t3phi_plot_all(self, ax) -> None:
        for i, o in enumerate(self.t3phidata):
            ax.plot(self.wl*1e6, unwrap_phase(o[11:-17]), label=self.tel_t3phi[i], linewidth=2)
            ax.set_ylim([-180, 180])
            ax.set_ylabel("cphase [deg]")
            ax.set_xlabel(r"wl [$\mu$ m]")

        ax.legend(loc=2)

    def vis24baseline_plot(self, ax, do_fit: bool = True) -> None:
        """ Plot the mean visibility for one certain wavelength and fit it with a gaussian and airy disk"""
        ax.plot(self.baseline_distances, self.mean_bin_vis2, ls='None', marker='o')

        # TODO: Implement fit here
        if do_fit:
            ...

        ax.set_xlabel(fr'uv-distance [m] at $\lambda_0$={trunc(self.mean_wl)} $\mu m$')
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

        # Compass for the directions
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

    def vis2_plot(self, axarr, err: bool = False) -> None:
        """Plots the squared visibilities"""
        # Sets the range for the squared visibility plots
        all_obs = [[],[],[],[],[],[]]

        # plots the squared visibility for different degrees and meters
        for i, o in enumerate(self.vis2data):
            # Sets the vis_dim if it is None
            # TODO: Rework this so it accounts for errors and really works as well
            # if self.vis_dim is None:
                # self.vis_dim = [0., np.mean(np.linalg.norm(o)*0.5]
            axis = axarr[0, i%6]
            axis.errorbar(self.wl*1e6, o, yerr=self.vis2err[i], marker='s',
                          label=fr"{self.tel_vis2[i]}; {np.sqrt(self.ucoords[i]**2+self.vcoords[i]**2)} m {pas} deg",
                          capsize=0., alpha=0.5)
            axis.set_ylim([self.vis_dim[0], self.vis_dim[1]])
            axis.set_ylabel("vis2")
            axis.set_xlabel("wl [micron]")
            all_obs[i%6].append(o)
            axis.legend(loc=2)

        # Plots the squared visibility errors
        for j in range(6):
            axis = axarr[0, j%6]

            if err:
                pas = (np.degrees(np.arctan2(self.vcoords[j], self.ucoords[j]))-90)*-1
                axis.errorbar(self.wl*1e6, np.nanmean(all_obs[j], 0), yerr=np.nanstd(all_obs[j], 0),
                              marker='s', capsize=0., alpha=0.9, color='k')

    def t3phi_plot(self, axarr, err=False) -> None:
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

            if err:
                axis.errorbar(self.wl*1e6, all_obs[j][0], yerr=np.std(all_obs[j], 0),\
                              marker='s', capsize=0., alpha=0.9, color='k', label=self.tel_t3phi[j])
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
        ax2.set_title(f"{model.name} FFT")
        ax2.set_xlabel("freq")
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])

        u, v = np.array([i[0] for i in uvcoords]), np.array([i[1] for i in uvcoords])
        ax2.scatter(u, v, s=5)

    def write_values(self) -> None:
        """"""
        with open(self.outname[:~7]+"_phase_values.txt", 'w') as f:
            for i in range(6):
                f.write(f"Vis2Data - {i}\n")
                f.write(str(self.vis2data[i]) + '\n')
                f.write("----------------------------------")
            for i in range(4):
                f.write(f"Unwrapped phase - {i}\n")
                f.write(str(unwrap_phase(self.t3phidata[i])) + '\n')
                f.write("----------------------------------")

    def close(self) -> None:
        """Finishing up the plot and then saving it to the designated folder"""
        plt.tight_layout()
        self.outname = self.dirname+'/'+self.fits_file.split('/')[-1]+'_qa.png'

        plt.savefig(self.outname, bbox_inches='tight')
        plt.close()


if __name__ == ('__main__'):
    ...
    # Tests
    # ------
    data_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514/coherent/lband/calib"
    # data_path = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs"
    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

    for i in subfolders:
        Plotter(i, True)

    # folder = "2021-10-15T07_20_19.AQUARIUS.rb_with_2021-10-15T06_50_56.AQUARIUS.rb_CALIBRATED"
    # Plotter(os.path.join(data_path, folder), [0., 0.15])
    # ------

    # Main process for shell usage
    # shell_main()

