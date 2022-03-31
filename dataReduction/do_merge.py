import os

from avg_oifits import oifits_patchwork

if __name__ == "__main__":
    base_folder = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514/"
    coherent_file = "coherent/nband/calib/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/TARGET_CAL_INT_0000.fits"
    incoherent_file = "incoherent/nband/calib/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/TARGET_CAL_INT_0000.fits"
    coherent_file = os.path.join(base_folder, coherent_file)
    incoherent_file = os.path.join(base_folder, incoherent_file)
    band = "nband" if "AQUARIUS" in coherent_file else "lband"
    outfile_dir = os.path.join(base_folder, "combined", band, )
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)
    outfile_path = os.path.join(outfile_dir, os.path.basename(coherent_file))
    print(outfile_dir)
    # oifits_patchwork([coherent_file, incoherent_file], outfile_path)

