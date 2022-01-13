import matplotlib.pyplot as plt

from fourier import FFT
from delta import Delta
from ring import Ring
from uniformdisk import UniformDisk
from gauss2d import Gauss2D
from opticallythinsphere import OpticallyThinSphere
from inclineddisk import InclinedDisk
from integraterings import IntegrateRings, Ring, blackbody_spec

def main(path, model):
    """Main function, executes code"""
    models = {"ring": Ring, "gauss": Gauss2D}
    model_array = models[model]().eval_model(1024, 10)
    fourier, uvcoords = FFT(model_array, 2048, path, 8e-06).fft_pipeline()[:2]

    return model_array, fourier, uvcoords

if __name__ == "__main__":
    main()
