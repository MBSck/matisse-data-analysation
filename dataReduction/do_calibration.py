import os

from glob import glob
from fluxcal import fluxcal

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

def do_reduction(folder_dir_tar, folder_dir_cal, mode="corrflux"):
    """Takes two folders and calibrates their contents together"""
    targets = glob(os.path.join(folder_dir_tar, "TARGET_RAW_INT*"))
    if not targets:
        targets = glob(os.path.join(folder_dir_tar, "CALIB_RAW_INT*"))
    targets.sort(key=lambda x: x[-8:])

    calibrators = glob(os.path.join(folder_dir_cal, "CALIB_RAW_INT*"))
    if not calibrators:
        raise RuntimeError("No 'CALIB_RAW_INT'-files found!")
    calibrators.sort(key=lambda x: x[-8:])

    # Formats the name of the new cal directory
    dir_name, time_sci, band = os.path.dirname(targets[0]).split('.')[:-1]
    dir_name = dir_name.split('/')[-1].replace("raw", "cal")
    time_cal = os.path.dirname(calibrators[0]).split('.')[-3]
    output_dir = os.path.join(base_path, "calib", '.'.join([dir_name, time_cal, band, time_sci, "rb"]))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, o in enumerate(targets):
        print(f"Calibrating {os.path.basename(o)} with "\
              f"{os.path.basename(calibrators[i])}")
        output_file = os.path.join(output_dir, f"TARGET_CAL_INT_000{i}.fits")
        fluxcal(o, calibrators[i], output_file,\
                CAL_DATABASE_PATHS, mode=mode, output_fig_dir=output_dir)

if __name__ == "__main__":
    # Gets the base paths
    base_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514/coherent/nband"
    folder_target = "mat_raw_estimates.2019-05-14T05_28_03.AQUARIUS.rb/"
    folder_cal = "mat_raw_estimates.2019-05-14T04_52_11.AQUARIUS.rb/"
    folder_target = os.path.join(base_path, folder_target)
    folder_cal = os.path.join(base_path, folder_cal)

    do_reduction(folder_target, folder_cal)

