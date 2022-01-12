#!/usr/bin/env python3

__author__ = "Jacob Isbell"

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pathlib import Path
from glob import glob
from scipy.optimize import curve_fit
from astropy.io import fits
<<<<<<< HEAD
from scipy.special import j0, j1        # Import of the Bessel functions of 0th and 1st order
from skimage.restoration import unwrap_phase
=======
from scipy.special import j0, j1                # Import of the Bessel functions of 0th and 1st order
>>>>>>> c174fdccfc4c57fc894fc501282d3ecb0dd6ff01

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
        fig, axarr = plt.subplots(3, 6, figsize=(16, 6))

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
        tel_names = hdu[2].data['tel_name']
        sta_name = hdu[2].data['sta_index']
        all_tels = ['A0', 'B2', 'C0', 'D1'] + ['K0', 'G1', 'D0', 'J3'] + ['A0', 'G1', 'J2', 'J3'] + ['UT1', 'UT2', 'UT3', 'UT4']    # Different baseline-configurations short-, medium-, large AT, UT
        all_stas = [1,  5, 13, 10] + [28, 18, 13, 24] + [1, 18, 23, 24] + [32, 33, 34, 35]                                          # 'sta_index'of telescopes
        telescopes = []
        for trio in loops:
            t1 = trio[0]#tel_names[np.where(sta_name == trio[0])[0]]
            t2 = trio[1]#tel_names[np.where(sta_name == trio[1])[0]]
            t3 = trio[2]#tel_names[np.where(sta_name == trio[2])[0]]
            telescopes.append('%s-%s-%s'%(all_tels[all_stas.index(t1)], all_tels[all_stas.index(t2)], all_tels[all_stas.index(t3)])) #[t1[0],t2[0],t3[0]])

        telnames_t3 = np.array(telescopes)

        # Sets the range for the squared visibility plots
        all_obs = [[],[],[],[],[],[]]

        # Plots the squared visibility for different degrees and meters
        for i, o in enumerate(vis2data):
            axis = axarr[0, i%6]
            axis.errorbar(wl*1e6, o, yerr=vis2err[i], marker='s', capsize=0., alpha=0.5)
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
<<<<<<< HEAD
            axis.errorbar(wl*1e6, unwrap_phase(o), yerr=t3phierr[i],marker='s',capsize=0.,alpha=0.25)
=======
            o_phase_unwrapped = np.arctan2(np.nanmean(np.sin(o*np.pi/180.0),axis=0),
                                           np.nanmean(np.cos(o*np.pi/180.0),axis=0))*180.0/np.pi
            print(o_phase_unwrapped)
            axis.errorbar(wl*1e6, o_phase_unwrapped, yerr=t3phierr[i], marker='s',capsize=0.,alpha=0.25)
>>>>>>> c174fdccfc4c57fc894fc501282d3ecb0dd6ff01
            axis.set_ylim([-180,180])
            axis.set_ylabel('cphase [deg]')
            axis.set_xlabel('wl [micron]')
            all_obs[i%4].append(list(o))

        for j in range(4):
            axis = axarr[1, j%4]
<<<<<<< HEAD
            # axis.errorbar(wl*1e6, all_obs[j][0], yerr=np.std(all_obs[j], 0), marker='s', capsize=0., alpha=0.9, color='k', label=telnames_t3[j])
            axis.legend([telnames_t3[j]], loc=2)
=======
            # Only takes a simple mean of the closure phases
            print(np.array(all_obs[j])[0])
            axis.errorbar(wl*1e6, np.nanmean(all_obs[j], 0), yerr=np.nanstd(unwrap_phase(np.array(all_obs[j][0])), 0), marker='s', capsize=0., alpha=0.9, color='k', label=telnames_t3[j])
            axis.legend(loc=2)
>>>>>>> c174fdccfc4c57fc894fc501282d3ecb0dd6ff01

        # Plots the uv coverage with a positional compass
        fx2.scatter(ucoord, vcoord)
        fx2.scatter(-ucoord, -vcoord)
        fx2.set_xlim([150, -150])
        fx2.set_ylim([-150, 150])
        # fx2.annotate("East to West", xytext=(-85, 135), fontsize=14)
        fx2.set_ylabel('v [m]')
        fx2.set_xlabel('u [m]')

        # Plot the mean visibility for one certain wavelength
        mean_vis4wl = [np.nanmean(i[5:115]) for i in vis2data]
        std_vis4wl = [np.nanstd(i[5:115]) for i in vis2data]
        baseline_distances = [np.sqrt(x**2+y**2) for x, y in zip(ucoord, vcoord)]
        mean_lambda = np.mean(wl)
        ax3.errorbar(baseline_distances, mean_vis4wl, yerr=std_vis4wl, ls='None', fmt='o')
        ax3.set_xlabel(fr'uv-distance [m] at $\lambda_0$={10.72} $\mu m$')
        ax3.set_ylabel(r'$\bar{V}$')

        # Plot around the mean wavelength for the different baselines
        wl_slice= [j for j in wl if (j >= mean_lambda-0.5e-06 and j <= mean_lambda+0.5e-06)]
        indicies_wl = []
        for i in wl_slice:
            indicies_wl.append(int(np.where(wl == i)[0]))
        si, ei = indicies_wl[0], indicies_wl[~0]

        for i in range(6):
            bx3.errorbar(wl[si:ei]*1e06, vis2data[i][si:ei], yerr=np.nanstd(vis2data[i][si:ei]), ls='None', fmt='o')
            bx3.set_xlabel(r'wl [micron]')
            bx3.set_ylabel('vis2')

        '''
        # Plots the squared visibility to the spatial frequency in Mlambda
        spat_freq = np.sqrt(np.square(ucoord)+np.square(vcoord))/3.6
        s = np.where(np.logical_and(wl>3.5e-6, wl<3.7e-6))[0][0]
        ex2.errorbar(spat_freq, vis2data[:, s], yerr=vis2err[:, s], marker='s', ls='none', color='firebrick')
        ex2.set_ylim([0, None])
        ex2.set_xlabel(r'Spat. Freq. M$\lambda$')
        ex2.set_ylabel('vis2 at 3.6um')
        '''

        # Fits the data
        if do_fit:
            scaling_rad2arc = 206265

            # Gaussian fit
            fwhm = 10/scaling_rad2arc/1000           # radians 
            xvals = np.linspace(30, 130)/3.6e-6      # np.linspace(np.min(spat_freq), np.max(spat_freq), 25)
            yvals = np.square(gaussian(xvals, fwhm))
            # print(yvals)
            ex2.plot(xvals/1e6, yvals, label='10mas Gaussian')

            # Airy-disk fit
            fwhm = 40/scaling_rad2arc/1000           # radians
            yvals = np.square(airy(xvals, fwhm))
            ex2.plot(xvals/1e6, yvals, label='%.1f mas Airy Disk'%(fwhm*206265*1000))
            # print(yvals)
            ex2.legend(loc='best')


        plt.tight_layout()
        outname = dirname+'/'+f.split('/')[-1]+'_qa.png'

        # plt.savefig(outname, bbinches='tight') # Bbinches is deprecated; Use bbox_inches
        plt.savefig(outname, bbox_inches='tight')
        plt.close()
        print(f"Done plotting {os.path.basename(Path(f))}")
        # plt.show()

if __name__ == ('__main__'):
    # Tests
    # ------
<<<<<<< HEAD
    data_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/calib_nband/UTs/"
    folders = [data_path + "2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T04_52_11.AQUARIUS.rb_CALIBRATED",
              data_path + "2019-05-14T04_52_11.AQUARIUS.rb_with_2019-05-14T06_12_59.AQUARIUS.rb_CALIBRATED",
              data_path + "2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T06_12_59.AQUARIUS.rb_CALIBRATED"]

    for i in folders:
        do_plot(i, [0., 0.15], do_fit=True)
        break
=======
    folder = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/calib_nband/UTs/2019-05-14T05_28_03.AQUARIUS.rb_with_2019-05-14T04_52_11.AQUARIUS.rb_CALIBRATED"
    #do_plot("2020-03-14T07_57_12.HAWAII-2RG.rb_with_2020-03-14T08_31_10.HAWAII-2RG.rb_CALIBRATED/", do_fit=True)
    #hdu = fits.open("/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/lband/mat_raw_estimates.2019-03-24T09_01_46.HAWAII-2RG.rb/TARGET_RAW_INT_0001.fits")
    #print(hdu[2].data["tel_name"])
    #print(hdu["oi_array"].data["tel_name"])

    do_plot(folder, vis_dim=[0., 0.6], do_fit=False)
>>>>>>> c174fdccfc4c57fc894fc501282d3ecb0dd6ff01
    # ------

    # Main process for shell usage
    # shell_main()

