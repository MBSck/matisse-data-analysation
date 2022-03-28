import os

from glob import glob
from fluxcal import fluxcal
from collections import deque

# inputfile_sci = 'path/to/raw/science/fits/file.fits'
# inputfile_cal = 'path/to/raw/calibrator/fits/file.fits'
# outputfile = 'path/to/calibrated/outputfile.fits'
# cal_database_dir = 'path/to/calibrator/database/folder/'
# cal_database_paths = [cal_database_dir+'vBoekelDatabase.fits',cal_database_dir+'calib_spec_db_v10.fits',cal_database_dir+'calib_spec_db_v10_supplement.fits']
# output_fig_dir = 'path/to/figure/folder/'
# fluxcal(inputfile_sci, inputfile_cal, outputfile, cal_database_paths, mode='corrflux',output_fig_dir=output_fig_dir)
#
# Arguments:
# inputfile_sci: path to the raw science oifits file.
# inputfile_cal: path to the raw calibrator oifits file.
# outputfile: path of the output calibrated file
# cal_database_paths: list of paths to the calibrator databases, e.g., [caldb1_path,caldb2_path]
# mode (optional):
#   'flux': calibrates total flux (incoherently processed oifits file expected)
#           results written in the OI_FLUX table (FLUXDATA column)
#    'corrflux': calibrates correlated flux (coherently processed oifits file expected)
#                results written in the OI_VIS table (VISAMP column)
#    'both': calibrates both total and correlated fluxes
# output_fig_dir (optional): if it is a valid path, the script will make a plot of the calibrator model spectrum there,
#                            deafult: '' (= no figure made)

# Datapath of the calib directories
CAL_DATABASE_DIR = os.path.join(os.getcwd(), "calib_spec_databases")
CAL_DATABASE_FILES = ['vBoekelDatabase.fits', 'calib_spec_db_v10.fits',
                      'calib_spec_db_v10_supplement.fits']
CAL_DATABASE_PATHS = [os.path.join(CAL_DATABASE_DIR, i) for i in CAL_DATABASE_FILES]

def single_reduction(folder_dir_tar: str, folder_dir_cal: str,
                     mode: str) -> int:
    """For documentation see 'do_reduction()'"""
    print(f"Calibrating {os.path.basename(folder_dir_tar)} with "\
          f"{os.path.basename(folder_dir_cal)}")
    dir_start = "TAR"

    targets = glob(os.path.join(folder_dir_tar, "TARGET_RAW_INT*"))
    if not targets:
        targets = glob(os.path.join(folder_dir_tar, "CALIB_RAW_INT*"))
        dir_start = "CAL"
    targets.sort(key=lambda x: x[-8:])

    calibrators = glob(os.path.join(folder_dir_cal, "CALIB_RAW_INT*"))
    if not calibrators:
        print("No 'CALIB_RAW_INT'-files found, Skipping!")
        print("------------------------------------------------------------")
        return -1
    calibrators.sort(key=lambda x: x[-8:])
    dir_start += "-CAL"

    # Formats the name of the new cal directory
    dir_name, time_sci, band = os.path.dirname(targets[0]).split('.')[:-1]
    dir_name = dir_name.split('/')[-1].replace("raw", "cal")
    time_cal = os.path.dirname(calibrators[0]).split('.')[-3]
    new_dir_name = '.'.join([dir_start, dir_name, time_sci, band, time_cal, "rb"])

    output_dir = os.path.join(base_path, "calib", new_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(len(targets), len(calibrators))

    for i, o in enumerate(targets):
        print("------------------------------------------------------------")
        print(f"Processing {os.path.basename(o)} with "\
              f"{os.path.basename(calibrators[i])}")
        output_file = os.path.join(output_dir, f"TARGET_CAL_INT_000{i}.fits")
        fluxcal(o, calibrators[i], output_file,\
                CAL_DATABASE_PATHS, mode=mode, output_fig_dir=output_dir)
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")


def do_reduction(base_path: str, folder_dir_tar: str = None,
                 folder_dir_cal: str = None, mode="corrflux") -> None:
    """Takes two folders and calibrates their contents together

    Parameters
    ----------
    base_path: str
        The path to multiple folders that need to be cross correlated. Will be
        skipped if folders for targets and calibrators are specified
    folder_dir_tar: str, optional
        A specific folder to be the target for calibration
    folder_dir_cal: str, optional
        A specific folder to be the calibrator for calibration
    mode: str, optional
        The mode of calibration. Either 'corrflux', 'flux' or 'both' depending
        if it is 'coherent' or 'incoherent'. Default mode is 'corrflux'
    """
    if folder_dir_tar is not None:
        single_reduction(folder_dir_tar, folder_dir_cal, mode)
    else:
        subdirs = glob(os.path.join(base_path, "*.rb"))

        # Rotates the list so it does not check itself
        subdirs_copy = deque(subdirs.copy())
        subdirs_copy.rotate(1)

        for i in subdirs:
            for j in subdirs_copy:
                if i != j:
                    single_reduction(i, j, mode)

def do_full_reduction(folder: str) -> None:
    """Does the full reduction for a full folder, being 'coherent',
    'incoherent' and for these 'lband' and 'nband', respectively"""
    modes, bands = {"coherent": "corrflux", "incoherent": "flux"},\
            ["lband", "nband"]

    for i, o in modes.items():
        path = os.path.join(folder, i)
        for j in bands:
            temp_path = os.path.join(path, j)
            print(f"Calibration of {temp_path}")
            print(f"with mode={o}")
            print("------------------------------------------------------------")
            do_reduction(temp_path, mode=o)

if __name__ == "__main__":
    base_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514"
    do_full_reduction(base_path)

