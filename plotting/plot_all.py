import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from astropy.io import fits
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

    for f in files[:]:
        print(f"Plotting {os.path.basename(Path(f))}")
        hdu = fits.open(f)
        fig, axarr = plt.subplots(2, 6, figsize=(16, 6))

        # Flattens the multidimensional arrays into 1D
        ax, bx, cx, dx, ex, fx = axarr[0].flatten()
        ax2, bx2, cx2, dx2, ex2, fx2 = axarr[1].flatten()

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


