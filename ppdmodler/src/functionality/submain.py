import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords, correspond_uv2model
from src.models import Gauss2D

# Shows the full np.arrays, takes ages to print the arrays
np.set_printoptions(threshold=sys.maxsize)


def main():
    file = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits"
    readout = ReadoutFits(file)
    wavelength = 9e-06
    print(readout.get_vis24wl(wavelength))
    uvcoords = readout.get_uvcoords()

    model = Gauss2D()
    model_vis  = model.eval_vis(128, 31.73645178, wavelength)
    model_axis_vis = model.axis_vis
    uvcoords, uv_ind = correspond_uv2model(model_vis, model_axis_vis, uvcoords, dis=True)
    print(uvcoords)
    # plt.imshow(model_vis)
    # plt.show()

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

if __name__ == "__main__":
    main()

