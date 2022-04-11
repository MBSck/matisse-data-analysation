import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords, correspond_uv2model
from src.models import Gauss2D, CompoundModel
from src.functionality.fourier import FFT

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    wavelength = 8e-6
    cp = CompoundModel(1500, 7900, 19, 140, wavelength)
    cp_model = cp.eval_model([1, 45], 5, 128)
    ax1.imshow(cp_model)

    ft, amp, phase = FFT(cp_model, wavelength).pipeline()
    ax2.imshow(abs(amp))
    plt.show()

if __name__ == "__main__":
    # print(comparefft2modvis(Gauss2D(), 8e-06))
    main()

