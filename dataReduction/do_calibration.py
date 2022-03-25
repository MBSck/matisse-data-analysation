import os

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

if __name__ == "__main__":
    base_path = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514/coherent/"
    inputfile_sci = "nband/mat_raw_estimates.2019-05-14T05_28_03.AQUARIUS.rb/TARGET_RAW_INT_0001.fits"
    inputfile_cal = "nband/mat_raw_estimates.2019-05-14T04_52_11.AQUARIUS.rb/CALIB_RAW_INT_0001.fits"

    output_dir = 
    outputfile = "TARGET_CAL_INT_0001.fits"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cal_database_dir = "calib_spec"
    cal_database_files = ['vBoekelDatabase.fits', 'calib_spec_db_v10.fits',
                          'calib_spec_db_v10_supplement.fits']
    cal_database_paths = [os.path.join(cal_database_dir, i) for i in cal_database_files]

    fluxcal(inputfile_sci, inputfile_cal, outputfile,\
            cal_database_paths, mode='corrflux',output_fig_dir=output_dir)

