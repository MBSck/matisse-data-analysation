import matplotlib.pyplot as plt

from functionality.fourier import FFT
from models.delta import Delta
from models.ring import Ring
from models.uniformdisk import UniformDisk
from models.gauss2d import Gauss2D
from models.opticallythinsphere import OpticallyThinSphere
from models.inclineddisk import InclinedDisk
from models.integraterings import IntegrateRings, Ring, blackbody_spec

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
    fig, (ax, bx, ax2, bx2) = plt.subplots(2, 2)

    ax1.imshow(int_uniform_disk_vis)
    ax1.set_title("Ring integrated uniform disk")
    ax2.imshow(uniform_disk_vis)
    ax2.set_title("Uniform disk analytical")
    plt.show()
    # plt.savefig("comparison_vis_differetn.png")


if __name__ == "__main__":
    main()
