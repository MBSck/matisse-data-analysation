import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from glob import glob
from scipy.optimize import curve_fit
from scipy.special import j0, j1        # Import of the Bessel functions of 0th and 1st order
import sys

def gaussian(spat_freq, D):
    """
    A gaussian fit described by the 0th-order Bessel function
        Parameters:
            spat_freq ():
            D ():
        Returns:
            gaussian_fit (ndarray): Gaussian fit
    """
    return np.exp( - np.square(np.pi* D * spat_freq) / (4* np.log(2) )  )

def airy(spat_freq, D):
    """
    An airy disk fit described by the 1st-order Bessel function
        Parameters:
            spat_freq ():
            D ():
        Returns:
            airy_disk_fit (ndarray): Airy disk fit
    """
    radial_dist = spat_freq * D
    return  2* j1( np.pi *radial_dist )/ radial_dist / np.pi

def do_plot(dirname, do_fit: bool = False) -> None:
    """
    Plots the
        Parameters:
            dirname:        Path to the directory, which files' are to be plotted
            do_fit (bool):  Bool that determines if fit is applied or not

        Returns:
            None
    """
    # This makes the plotter function by itself with shell input
    try:
            dirname = sys.argv[1]
    except:
        print("No shell input given. Proceeding with arguments")

    # Sorts the 'CAL_INT*.fits'-files
    files = np.sort( glob(dirname + '/*CAL_INT*.fits')  )
    for f in files[:]:
        hdu = fits.open(f)
        fig, axarr = plt.subplots(2,6,figsize=(16,6))

        # Flattens the multidimensional arrays into 1D
        ax,bx,cx,dx,ex,fx = axarr[0].flatten()
        ax2,bx2,cx2,dx2,ex2,fx2 = axarr[1].flatten()

        # Gets the data from the '.fits'-file
        vis2data= hdu['oi_vis2'].data['vis2data']
        vis2err = hdu['oi_vis2'].data['vis2err']
        ucoord = hdu['oi_vis2'].data['ucoord']
        vcoord = hdu['oi_vis2'].data['vcoord']
        wl = hdu['oi_wavelength'].data['eff_wave']
        t3phi = hdu['oi_t3'].data['t3phi']          # Use 't3phi', closure phase, as 't3amp' carries no real info
        t3phierr = hdu['oi_t3'].data['t3phierr']

        # Gets the baseline configuration of the telescopes
        loops = hdu['OI_T3'].data['sta_index']  # 'sta_index' short for station index, describing the telescope-baseline relationship
        tel_names = hdu[2].data['tel_name']
        sta_name = hdu[2].data['sta_index']
        all_tels = ['A0', 'B2', 'C0', 'D1'] + ['K0', 'G1', 'D0', 'J3'] + [] + ['UT1', 'UT2', 'UT3', 'UT4']  # Different baseline-configurations short AT, , , UT
        all_stas = [1,  5, 13, 10] + [28, 18, 13, 24] + [] + [32, 33, 34, 35]                               # 'sta_index'of telescopes
        telescopes = []
        for trio in loops:
            t1 = trio[0]#tel_names[np.where(sta_name == trio[0])[0]]
            t2 = trio[1]#tel_names[np.where(sta_name == trio[1])[0]]
            t3 = trio[2]#tel_names[np.where(sta_name == trio[2])[0]]
            telescopes.append('%s-%s-%s'%(all_tels[all_stas.index(t1)], all_tels[all_stas.index(t2)], all_tels[all_stas.index(t3)])) #[t1[0],t2[0],t3[0]])

        telnames_t3 = np.array(telescopes)

        # Sets the range for the squared visibility plots
        all_obs = [[],[],[],[],[],[]]
        for b in range(len(vis2data)):
            axis = axarr[0, b%6  ]
            axis.errorbar(wl * 1e6, vis2data[b],yerr=vis2err[b],marker='s',capsize=0.,alpha=0.5)
            axis.set_ylim([0,2.])  # 0.045 formerly
            axis.set_ylabel('vis2')
            axis.set_xlabel('wl [micron]')
            all_obs[b%6].append(vis2data[b])

        # Plots the squared visibility for different degrees and metres
        for b in range(6):
            axis = axarr[0, b%6  ]
            pas = (np.degrees(np.arctan2(vcoord[b],ucoord[b])) - 90) * -1
            axis.errorbar(wl * 1e6, np.nanmean(all_obs[b],0),yerr=np.nanstd(all_obs[b],0),marker='s',capsize=0.,alpha=0.9,color='k',label='%.1f m %.1f deg'%(np.sqrt(ucoord[b]**2 + vcoord[b]**2),pas   ))
            axis.legend(loc=2)

        # Plots the closure phase
        all_obs = [[],[],[],[]]
        for b in range(len(t3phi)):
            axis = axarr[1, b%4  ]
            axis.errorbar(wl * 1e6, t3phi[b],yerr=t3phierr[b],marker='s',capsize=0.,alpha=0.25)
            axis.set_ylim([-180,180])
            axis.set_ylabel('cphase [deg]')
            axis.set_xlabel('wl [micron]')
            all_obs[b%4].append(t3phi[b])
        for b in range(4):
            axis = axarr[1, b%4  ]
            axis.errorbar(wl * 1e6, np.nanmean(all_obs[b],0),yerr=np.nanstd(all_obs[b],0),marker='s',capsize=0.,alpha=0.9,color='k',label=telnames_t3[b])
            axis.legend(loc=2)

        # Plots the uv coverage
        fx2.scatter(ucoord, vcoord)
        fx2.scatter(-ucoord, -vcoord)
        fx2.set_xlim([140,-140])
        fx2.set_ylim([-140,140])
        fx2.set_ylabel('v [m]')
        fx2.set_xlabel('u [m]')

        # Plots the
        spat_freq = np.sqrt(np.square(ucoord) + np.square(vcoord) ) / 3.6
        s = np.where( np.logical_and(wl>3.5e-6, wl<3.7e-6)   )[0][0]
        ex2.errorbar(spat_freq, vis2data[:,s],yerr=vis2err[:,s],marker='s',ls='none',color='firebrick'    )
        ex2.set_ylim([0,None])
        ex2.set_xlabel(r'Spat. Freq. M$\lambda$')
        ex2.set_ylabel('vis2 at 3.6um')

        # Fits the data
        if do_fit:
            # Gaussian fit
            fwhm = 10 / 206265 / 1000           #radians
            D = fwhm / .832
            xvals = np.linspace(30, 130)/3.6e-6 #np.linspace(np.min(spat_freq), np.max(spat_freq), 25)
            yvals = np.square(gaussian(xvals , D))
            print(yvals)
            ex2.plot(xvals / 1e6, yvals, label='10mas Gaussian')

            # Airy-disk fit
            fwhm = 40 / 206265 / 1000           #radians
            D = fwhm
            yvals = np.square(airy(xvals , D))
            ex2.plot(xvals / 1e6, yvals, label='%.1f mas Airy Disk'%(fwhm * 206265 * 1000))
            print(yvals)
            ex2.legend(loc='best')


        plt.tight_layout()
        outname = dirname + '/'+f.split('/')[-1] + '_qa.png'
        #plt.savefig(outname, bbinches='tight') # Bbinches is outdated and will become an error, use bbox_inches instead
        plt.savefig(outname, bbox_inches='tight')
        plt.close()
        #plt.show()

if __name__ == ('__main__'):
    #do_plot("2020-03-14T07_57_12.HAWAII-2RG.rb_with_2020-03-14T08_31_10.HAWAII-2RG.rb_CALIBRATED/", do_fit=True)
    #hdu = fits.open("/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/lband/mat_raw_estimates.2019-03-24T09_01_46.HAWAII-2RG.rb/TARGET_RAW_INT_0001.fits")
    #print(hdu[2].data["tel_name"])
    #print(hdu["oi_array"].data["tel_name"])

    do_plot("/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/calib/20190324/calTarSTD1", do_fit=True)
