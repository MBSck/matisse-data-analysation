import os

from glob import glob
from fluxcal import fluxcal
from collections import deque

"""Wrapper for Jozsef Varga' script fluxcal"""

# Datapath of the calib directories, global variables
CAL_DATABASE_DIR = os.path.join(os.getcwd(), "calib_spec_databases")
CAL_DATABASE_FILES = ['vBoekelDatabase.fits', 'calib_spec_db_v10.fits',
                      'calib_spec_db_v10_supplement.fits']
CAL_DATABASE_PATHS = [os.path.join(CAL_DATABASE_DIR, i) for i in CAL_DATABASE_FILES]

def get_folder_path(dir_start: str, target: str,
                    calibrator: str) -> str:
    """Makes the folder names of the new folder for the calibrated files and
    returns the 'output_dir'"""
    working_dir = os.path.dirname(os.path.dirname(target))

    # Formats the name of the new cal directory
    dir_name, time_sci, band = os.path.dirname(target).split('.')[:-1]
    dir_name = dir_name.split('/')[-1].replace("raw", "cal")
    time_cal = os.path.dirname(calibrator).split('.')[-3]
    new_dir_name = '.'.join([dir_start, dir_name, time_sci, band, time_cal, "rb"])

    return os.path.join(working_dir, "calib", new_dir_name)

def single_reduction(folder_dir_tar: str, folder_dir_cal: str,
                     mode: str) -> int:
    """The calibration for a target and a calibrator folder

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
    print(f"Calibrating {os.path.basename(folder_dir_tar)} with "\
          f"{os.path.basename(folder_dir_cal)}")
    dir_start = "TAR"

    targets = glob(os.path.join(folder_dir_tar, "TARGET_RAW_INT*"))
    if not targets:
        targets = glob(os.path.join(folder_dir_tar, "CALIB_RAW_INT*"))
        dir_start = "CAL"

    targets.sort(key=lambda x: x[-8:])

    calibrators = glob(os.path.join(folder_dir_cal, "CALIB_RAW_INT*"))
    calibrators.sort(key=lambda x: x[-8:])
    dir_start += "-CAL"

    if not calibrators:
        print("No 'CALIB_RAW_INT'-files found, probably target. Skipping!")
        print("------------------------------------------------------------")
        return -1

    if len(targets) != len(calibrators):
        print("#'TARGET_RAW_INT'-files != #'CALIB_RAW_INT'-files. Skipping!")
        print("------------------------------------------------------------")
        return -2

    output_dir = get_folder_path(dir_start, targets[0], calibrators[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, o in enumerate(targets):
        print("------------------------------------------------------------")
        print(f"Processing {os.path.basename(o)} with "\
              f"{os.path.basename(calibrators[i])}")
        output_file = os.path.join(output_dir, f"TARGET_CAL_INT_000{i+1}.fits")

        fluxcal(o, calibrators[i], output_file,\
                CAL_DATABASE_PATHS, mode=mode, output_fig_dir=output_dir)
    print(f"Done calibrating {os.path.basename(folder_dir_tar)} with "\
          f"{os.path.basename(folder_dir_cal)}")
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

def do_full_reduction(folder: str, both: bool = True,
                      lband: bool = False) -> None:
    """Does the full reduction for a full folder, being 'coherent',
    'incoherent' and for these 'lband' and 'nband', respectively

    Parameters
    ----------
    folder: str
        The main folder to be calibrated
    both: bool
        If both nband and lband should be calibrated
    lband: bool
        If 'both=False' and this is True, then lband will be calibrated, if
        'both=False" and this is False, then nband will be calibrated
    """
    modes, bands = {"coherent": "corrflux", "incoherent": "flux"},\
            ["lband", "nband"]

    if both:
        for i, o in modes.items():
            path = os.path.join(folder, i)
            for j in bands:
                temp_path = os.path.join(path, j)
                print(f"Calibration of {temp_path}")
                print(f"with mode={o}")
                print("------------------------------------------------------------")
                do_reduction(temp_path, mode=o)
    else:
        for i, o in modes.items():
            band = "lband" if lband else "nband"
            path = os.path.join(folder, i, band)
            print(f"Calibration of {path}")
            print(f"with mode={o}")
            print("------------------------------------------------------------")
            do_reduction(path, mode=o)


if __name__ == "__main__":
    # base_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514"
    # do_full_reduction(base_path, both=True)

