import matplotlib.pyplot as plt

from modelling.functionality.fourier import FFT
from modelling.models.delta import Delta
from modelling.models.uniformdisk import UniformDisk
from modelling.models.gauss2d import Gauss2D
from modelling.models.opticallythinsphere import OpticallyThinSphere
from modelling.models.inclineddisk import InclinedDisk
from modelling.models.integraterings import IntegrateRings

from modelling.functionality.utilities import compare_arrays

def main(model):
    """Main function, executes code"""
    fourier = FFT(model.eval_model(512, 256.1), 512, "./assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits", 8e-06).fft_pipeline()[0]

    vis_analytically = model.eval_vis(512, 256.1, 8e-06)
    plt.imshow(vis_analytically)
    plt.show()

    print(compare_arrays(fourier, vis_analytically))

if __name__ == "__main__":
    main((Gauss2D()))
