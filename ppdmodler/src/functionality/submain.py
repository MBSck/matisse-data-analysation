import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords, correspond_uv2model
from src.models import Gauss2D
from src.functionality.fourier import FFT

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

def comparefft2modvis(model, wavelength):
    model_mod = model.eval_model(128, 10.)
    model_vis = model.eval_vis(128, 10., wavelength)
    ft = FFT(model_mod, 0, wavelength).do_fft2()
    return [np.mean(np.array([[ np.linalg.norm(i-j) for j in model_vis[x]] for i in ft[x]])) for x in range(128)], \
            (np.mean(np.abs(model_vis)), np.mean(np.abs(ft)))

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sampling = 1024
    gauss_mod = Gauss2D().eval_model([100., 1.], 10., sampling)
    ax1.imshow(gauss_mod)

    ft, amp, phase = FFT(gauss_mod, 8e-6).pipeline()
    print(amp, phase)
    ft_size = len(ft)
    print(amp[ft_size//2][ft_size//2-3:ft_size//2+3])
    ax2.imshow(np.log(abs(ft)))
    plt.show()

    '''
    file = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(file)
    wavelength = readout.get_wl()[80]
    print(wavelength)
    print(readout.get_vis24wl(80))
    uvcoords = readout.get_uvcoords()
    print(uvcoords)

    model = Gauss2D()
    model_vis  = model.eval_vis(128, 35.5124433, wavelength, None)
    print(model_vis)
    model_axis_vis = model.axis_vis
    uvcoords, uv_ind = correspond_uv2model(model_vis, model_axis_vis, uvcoords, dis=True)
    print(uvcoords, "The corresponded vis2data")
    # plt.imshow(model_vis)
    # plt.show()
    '''

    '''
    # Set up the parameters of the problem.
    ndim, nsamples = 3, 50000

    # Generate some fake data.
    np.random.seed(42)
    data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape([4 * nsamples // 5, ndim])
    data2 = (4*np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples // 5).reshape([nsamples // 5, ndim]))
    data = np.vstack([data1, data2])

    # Plot it.
    fig =  corner.corner(data, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"],
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12})
    plt.show()
    '''

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()

