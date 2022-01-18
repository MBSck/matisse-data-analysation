__author__ = "Jacob Isbell"

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


def shell_main():
    """
    This function's sole purpose is to enable the plotter to work in the shell
    """
    try:
        script, dirname, vis_dim0, vis_dim1 = sys.argv
    except:
        print("Usage: python3 myplotter.py /path/to/target/data/dir/ vis_dim[0] vis_dim[1]")
        sys.exit(1)

    vis_dim = [float(vis_dim0), float(vis_dim1)]
    do_plot(dirname=dirname, vis_dim=vis_dim, do_fit=False)

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


def do_plot(dirname: str, vis_dim: list, do_fit: bool = False) -> None:
    """Plots the

    Parameters
    ----------
    dirname: Path[str]
        Path to the directory, which files' are to be plotted
    do_fit: bool
        Bool that determines if fit is applied or not

    Returns
    -------
    None
    """
    # Sorts the 'CAL_INT*.fits'-files
    files = np.sort(glob(dirname + '/*CAL_INT*.fits'))

    # Checks if no files are found
    if files is None:
        print("No files found! Check input path")
        sys,exit(1)

    for f in files[:]:
        print(f"Plotting {os.path.basename(Path(f))}")
        hdu = fits.open(f)
        fig, axarr = plt.subplots(3, 6, figsize=(20, 10))

        # Flattens the multidimensional arrays into 1D
        ax, bx, cx, dx, ex, fx, = axarr[0].flatten()
        ax2, bx2, cx2, dx2, ex2, fx2 = axarr[1].flatten()
        ax3, bx3, cx3, dx3, ex3, fx3 = axarr[2].flatten()

        # Gets the data from the '.fits'-file
        vis2data = hdu['oi_vis2'].data['vis2data'][:6]
        vis2err = hdu['oi_vis2'].data['vis2err'][:6]
        ucoord = hdu['oi_vis2'].data['ucoord'][:6]
        vcoord = hdu['oi_vis2'].data['vcoord'][:6]
        wl = hdu['oi_wavelength'].data['eff_wave']

        # Use 't3phi', closure phase, as 't3amp' carries no real info
        t3phi = hdu['oi_t3'].data['t3phi'][:4]
        t3phierr = hdu['oi_t3'].data['t3phierr'][:4]

        # Gets the baseline configuration of the telescopes
        loops = hdu['OI_T3'].data['sta_index']  # 'sta_index' short for station index, describing the telescope-baseline relationship
        vis_loops = hdu['oi_vis2'].data['sta_index']

        tel_names = hdu[2].data['tel_name']
        sta_name = hdu[2].data['sta_index']
        all_tels = ['A0', 'B2', 'C0', 'D1'] + ['K0', 'G1', 'D0', 'J3'] + ['A0', 'G1', 'J2', 'J3'] + ['UT1', 'UT2', 'UT3', 'UT4']    # Different baseline-configurations short-, medium-, large AT, UT
        all_stas = [1,  5, 13, 10] + [28, 18, 13, 24] + [1, 18, 23, 24] + [32, 33, 34, 35]                                          # 'sta_index'of telescopes
        telescopes = []
        for trio in loops:
            # tel_names[np.where(sta_name == trio[2])[0]]
            t1, t2, t3 = trio
            telescopes.append('%s-%s-%s'%(all_tels[all_stas.index(t1)], all_tels[all_stas.index(t2)], all_tels[all_stas.index(t3)])) #[t1[0],t2[0],t3[0]])
        telnames_t3 = np.array(telescopes)

        telescopes = []
        for duo in vis_loops:
            t1, t2 = duo
            telescopes.append(f"{all_tels[all_stas.index(t1)]}-{all_tels[all_stas.index(t2)]}")
        telnames_vis2 = np.array(telescopes)

        # Sets the range for the squared visibility plots
        all_obs = [[],[],[],[],[],[]]

        # Plots the squared visibility for different degrees and meters
        for i, o in enumerate(vis2data):
            axis = axarr[0, i%6]
            axis.errorbar(wl*1e6, o, yerr=vis2err[i], marker='s', label=telnames_vis2[i], capsize=0., alpha=0.5)
            axis.set_ylim([vis_dim[0], vis_dim[1]])
            axis.set_ylabel('vis2')
            axis.set_xlabel('wl [micron]')
            all_obs[i%6].append(o)

        # Plots the squared visibility for different degrees and metres
        for j in range(6):
            axis = axarr[0, j%6]
            pas = (np.degrees(np.arctan2(vcoord[j],ucoord[j]))-90)*-1
            axis.errorbar(wl*1e6, np.nanmean(all_obs[j], 0), yerr=np.nanstd(all_obs[j], 0), marker='s', capsize=0., alpha=0.9, color='k', label='%.1f m %.1f deg'%(np.sqrt(ucoord[j]**2+vcoord[j]**2), pas))
            axis.legend(loc=2)

        # Plots the closure phase
        all_obs = [[],[],[],[]]
        for i, o in enumerate(t3phi):
            axis = axarr[1, i%4]
            axis.errorbar(wl*1e6, unwrap_phase(o), yerr=t3phierr[i],marker='s',capsize=0.,alpha=0.25)
            axis.set_ylim([-180,180])
            axis.set_ylabel('cphase [deg]')
            axis.set_xlabel('wl [micron]')
            all_obs[i%4].append(list(o))

        for j in range(4):
            axis = axarr[1, j%4]
            # axis.errorbar(wl*1e6, all_obs[j][0], yerr=np.std(all_obs[j], 0), marker='s', capsize=0., alpha=0.9, color='k', label=telnames_t3[j])
            axis.legend([telnames_t3[j]], loc=2)

        '''
        # Plot the mean visibility for one certain wavelength
        mean_vis4wl = [np.nanmean(i[5:115]) for i in vis2data]
        std_vis4wl = [np.nanstd(i[5:115]) for i in vis2data]
        baseline_distances = [np.sqrt(x**2+y**2) for x, y in zip(ucoord, vcoord)]
        bx3.errorbar(baseline_distances, mean_vis4wl, yerr=std_vis4wl, ls='None', fmt='o')
        bx3.set_xlabel(fr'uv-distance [m] at $\lambda_0$={10.72} $\mu m$')
        bx3.set_ylabel(r'$\bar{V}$')
        '''

        # Plot waterfall with the mean wavelength for the different baselines
        mean_lambda = np.mean(wl)
        wl_slice= [j for j in wl if (j >= mean_lambda-0.5e-06 and j <= mean_lambda+0.5e-06)]
        indicies_wl = []
        for i in wl_slice:
            indicies_wl.append(int(np.where(wl == i)[0]))
        si, ei = indicies_wl[0]-5, indicies_wl[~0]-5

        for i in range(6):
            fx2.errorbar(wl[si:ei]*1e06, vis2data[i][si:ei], yerr=np.nanstd(vis2data[i][si:ei]),label=telnames_vis2[i], ls='None', fmt='o')
            fx2.set_xlabel(r'wl [micron]')
            fx2.set_ylabel('vis2')
            fx2.legend(loc='best')

        # Plot the mean visibility for one certain wavelength and fit it with a gaussian and airy disk
        mean_bin_vis2 = [np.nanmean(i[si:ei]) for i in vis2data]
        std_bin_vis2 = [np.nanmean(i[si:ei]) for i in vis2data]
        baseline_distances = [np.sqrt(x**2+y**2) for x, y in zip(ucoord, vcoord)]
        ex2.errorbar(baseline_distances, mean_bin_vis2, yerr=std_bin_vis2, ls='None', fmt='o')

        # Fits the data
        if do_fit:
            scaling_rad2arc = 206265
            # Gaussian fit
            fwhm = 1/scaling_rad2arc/1000           # radians
            xvals = np.linspace(50, 3*150)/3.6e-6      # np.linspace(np.min(spat_freq), np.max(spat_freq), 25)
            fitted_model= np.square(gaussian(xvals, fwhm))
            ex2.plot(xvals/1e6, fitted_model*0.15, label='Gaussian %.1f"'%(fwhm*scaling_rad2arc*1000))

            # Airy-disk fit
            fwhm = 3/scaling_rad2arc/1000           # radians
            fitted_model = np.square(airy(xvals, fwhm))
            ex2.plot(xvals/1e6, fitted_model*0.15, label='Airy Disk %.1f"'%(fwhm*scaling_rad2arc*1000))
            ex2.set_ylim([0, 0.175])
            ex2.legend(loc='best')

        ex2.set_xlabel(fr'uv-distance [m] at $\lambda_0$={10.72} $\mu m$')
        ex2.set_ylabel(r'$\bar{V}$')

        # Plots the uv coverage with a positional compass
        ax3.scatter(ucoord, vcoord)
        ax3.scatter(-ucoord, -vcoord)
        ax3.set_xlim([150, -150])
        ax3.set_ylim([-150, 150])
        ax3.set_ylabel('v [m]')
        ax3.set_xlabel('u [m]')

        # Primitive legend or the directions of the uv-plot, if from source or from telescope seen
        cardinal_vectors = [(0,1), (0,-1), (1,0), (-1,0)]   # north, south, east, west
        cardinal_colors  = ['black', 'green', 'blue', 'red']
        cardinal_directions = ['N', 'S', 'W', 'E']
        arrow_size, head_size = 40, 10
        x, y = (-85, 85)

        for vector, color, direction in zip(cardinal_vectors, cardinal_colors, cardinal_directions):
            dx, dy = vector[0]*arrow_size, vector[1]*arrow_size
            if vector[0] == 0:
                ax3.text(x-dx-5, y+dy, direction)
            if vector[1] == 0:
                ax3.text(x-dx, y+dy+5, direction)
            arrow_args = {"length_includes_head": True, "head_width": head_size, "head_length": head_size, \
                                  "width": 1, "fc": color, "ec": color}
            ax3.arrow(x, y, dx, dy, **arrow_args)

        # Plots the model fits and their fft of the uv-coords
        gauss_img, gauss_ft, uvcoords = main(f, "gauss")
        ring_img, ring_ft, uvcoords = main(f, "ring")

        # Plots the gaussian model
        bx3.imshow(gauss_img)
        bx3.set_title(f'Model Gauss 10"')
        bx3.set_xlabel(f"resolution [px] 1024, zero padding 2048")
        bx3.axes.get_xaxis().set_ticks([])
        bx3.axes.get_yaxis().set_ticks([])


        cx3.imshow(np.log(abs(gauss_ft)), interpolation='none', extent=[-0.5, 0.5, -0.5, 0.5])
        cx3.set_title("Gauss FFT")
        cx3.set_xlabel("freq")
        cx3.axes.get_xaxis().set_ticks([])
        cx3.axes.get_yaxis().set_ticks([])

        u, v = np.array([i[0] for i in uvcoords]), np.array([i[1] for i in uvcoords])
        cx3.scatter(u, v, s=5)

        # Plots the ring model
        dx3.imshow(ring_img)
        dx3.set_title(f'Model Ring 10"')
        dx3.set_xlabel(f"resolution [px] 1024, zero padding 2048")
        dx3.axes.get_xaxis().set_ticks([])
        dx3.axes.get_yaxis().set_ticks([])

        ex3.imshow(np.log(abs(ring_ft)), interpolation='none', extent=[-0.5, 0.5, -0.5, 0.5])
        ex3.set_title("Ring FFT")
        ex3.set_xlabel("freq")
        ex3.axes.get_xaxis().set_ticks([])
        ex3.axes.get_yaxis().set_ticks([])

        u, v = np.array([i[0] for i in uvcoords]), np.array([i[1] for i in uvcoords])
        ex3.scatter(u, v, s=5)

        # Finishing up
        plt.tight_layout()
        outname = dirname+'/'+f.split('/')[-1]+'_qa.png'

        plt.savefig(outname, bbox_inches='tight')
        plt.close()
        print(f"Done plotting {os.path.basename(Path(f))}")

        with open(outname[:~7]+"_phase_values.txt", 'w') as f:
            for i in range(4):
                f.write(str(unwrap_phase(t3phi[i])) + '\n')

if __name__ == ('__main__'):
    # Tests
    # ------
    data_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/calib_nband/UTs/"
    folders = [data_path + "2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T04_52_11.AQUARIUS.rb_CALIBRATED",
              data_path + "2019-05-14T04_52_11.AQUARIUS.rb_with_2019-05-14T06_12_59.AQUARIUS.rb_CALIBRATED",
              data_path + "2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T06_12_59.AQUARIUS.rb_CALIBRATED"]

    for i in folders:
        do_plot(i, [0., 0.15], do_fit=True)
    folder = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/calib_nband/UTs/2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T04_52_11.AQUARIUS.rb_CALIBRATED"
    # ------

    # Main process for shell usage
    # shell_main()

