import matplotlib.pyplot as plt

from modelling.functionality.fourier import FFT
from modelling.models.delta import Delta
from modelling.models.uniformdisk import UniformDisk
from modelling.models.gauss2d import Gauss2D
from modelling.models.opticallythinsphere import OpticallyThinSphere
from modelling.models.inclineddisk import InclinedDisk
from modelling.models.integraterings import IntegrateRings, Ring, blackbody_spec

from modelling.functionality.utilities import compare_arrays

def main():
    """Main function, executes code"""
    models = [Ring, UniformDisk]
    '''
    for i in models:
        fourier = FFT(i().eval_model(512, 256.1), 512, "./assets/TARGET_CAL_INT_0001bcd_calibratedTEST.fits", 8e-06).fft_pipeline()[0]

        vis_analytically = i().eval_vis(512, 256.1, 8e-06)
        plt.imshow(vis_analytically)
        plt.show()

        print(compare_arrays(fourier, vis_analytically))
    '''

    inte = IntegrateRings()
    int_uniform_disk_vis = inte.integrate_rings_vis(512, 1, 50, 0.55, 6000, 8e-06)
    u = models[1]()
    uniform_disk_vis = u.eval_vis(512, 50, 8e-06)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(int_uniform_disk_vis)
    ax1.set_title("Ring integrated uniform disk")
    ax2.imshow(uniform_disk_vis)
    ax2.set_title("Uniform disk analytical")
    plt.show()
    # plt.savefig("comparison_vis_differetn.png")


if __name__ == "__main__":
    main()
