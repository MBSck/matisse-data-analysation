#!/usr/bin/env python3

import os
import glob

author = "Marten Scheuck"

"""Credit to Jozsef Varga, who's script this is based upon. DRS for both flux
and vis2"""

SPECTRALBINNING_L = '5' #DRS option (tried value = 5 and 10 - matisse_redM), default for LR: '1'
SPECTRALBINNING_N = '7' #21 #DRS option, default for LR: '7' (tried value = 49 - matisse_redM and some matisse_red6)

DRS_MODES = ["coherent", "incoherent"]

#for coherent visibilities: '/corrFlux=FALSE/useOpdMod=FALSE/coherentAlgo=2/compensate="[pb,rb,nl,if,bp,od]"/'
PARAM_L_LIST = ['/spectralBinning='+SPECTRALBINNING_L+'/corrFlux=TRUE/useOpdMod=FALSE/coherentAlgo=2/compensate="[pb,rb,nl,if,bp,od]"/',
               '/spectralBinning='+SPECTRALBINNING_L+'/compensate="pb,rb,nl,if,bp,od"/']

PARAM_N_LIST_AT = ['/replaceTel=3/corrFlux=TRUE/useOpdMod=TRUE/coherentAlgo=2/spectralBinning='+SPECTRALBINNING_N,
                  '/replaceTel=3/spectralBinning='+SPECTRALBINNING_N]
PARAM_N_LIST_UT = ['/replaceTel=0/corrFlux=TRUE/useOpdMod=TRUE/coherentAlgo=2/spectralBinning='+SPECTRALBINNING_N,
                  '/replaceTel=0/spectralBinning='+SPECTRALBINNING_N]


def run_pipeline(input_path: str, output_path: str, data: dict, do_L: bool = True,
                 do_plot: bool = False, do_reduction: bool = True):
    # Set params and other values
    skip_L, skip_N =  int(not do_L), int(do_L)
    night, tpl_start, tel, dil, din = map(data.get, data)

    # ----------run the pipeline-------------------------------
    for i, o in enumerate(DRS_MODES):
        resdir = os.path.join(output_path, o, night, tpl_start.replace(':', '_'))

        if not os.path.exists(resdir):
            os.makedirs(resdir)

        if do_reduction:
            # Clears the old *.sof'-files
            if do_L:
                resdir_l = glob.glob(os.path.join(resdir, "Iter1/*HAWAII*/"))
                if resdir_l:
                    shutil.rmtree(resdir_l[0])
                soffiles = glob.glob(os.path.join(resdir, "/Iter1/*HAWAII*.sof*"))

                if soffiles:
                    for file in soffiles:
                        os.remove(file)
            else:
                resdir_n = glob.glob(os.path.join(resdir, "/Iter1/*AQUARIUS*/"))
                if resdir_n:
                    shutil.rmtree(resdir_n[0])
                soffiles = glob.glob(os.path.join(resdir, "/Iter1/*AQUARIUS*.sof*"))
                if soffiles:
                    for file in soffiles:
                        os.remove(file)

            # Sets the correct params for the array
            if tel == 'UTs':
                PARAM_N_LIST = PARAMN_LIST_UT
            if tel == 'ATs':
                PARAM_N_LIST = PARAMN_LIST_AT

            # Set the correct binning 
            if 'HIGH' in din:
                for k, l in enumerate(PARAM_N_LIST):
                    PARAM_N_LIST[k] = l.replace('spectralBinning=7','spectralBinning=49')
            else:
                for m, n in enumerate(PARAM_N_LIST):
                    PARAM_N_LIST[m] = n.replace('spectralBinning=49','spectralBinning=7')

            # first try to find calibration files in RAWDIR+'/calibration_files/'
            if os.path.exists(os.path.join(rawdir, "calibration_files")):
                res = mat_autoPipeline.mat_autoPipeline(dirRaw=rawdir, dirResult=resdir, dirCalib=os.patj.join(rawdir, "calibration_files"), nbCore=6, tplstartsel=tpl_start,
                                              resol='', paramL=PARAM_L_LIST[i], paramN=PARAMN_LIST[i], overwrite=0, maxIter=1,
                                              skipL=skip_L, skipN=skip_N)
                if res == 2: #if missing calibration
                    # if calibration files were not found, then use general calibration directory (CALIBDIR)
                    res = mat_autoPipeline.mat_autoPipeline(dirRaw=rawdir, dirResult=resdir, dirCalib=calibdir, nbCore=6, tplstartsel=tpl_start,
                                              resol='', paramL=PARAM_L_LIST[i], paramN=PARAMN_LIST[i], overwrite=0, maxIter=1,
                                              skipL=skip_L, skipN=skip_N)
            else:
                # if there is no calibration directory within the night folder, then use general calibration directory (CALIBDIR)
                res = mat_autoPipeline.mat_autoPipeline(dirRaw=rawdir, dirResult=resdir, dirCalib=rawdir, nbCore=6, tplstartsel=tpl_start,
                                              resol='', paramL=PARAM_L_LIST[i], paramN=PARAM_N_LIST[i], overwrite=0, maxIter=1,
                                              skipL=skip_L, skipN=skip_N)


if __name__ == "__main__":
    datadir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/RAW/20190514"
    resdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/test_20190514"
    data = {"night": "2019-05-13", "tpl_start": "2019-05-14T05:28:03",
            "tel": "UTs", "dil": "LOW", "din": "LOW"}
    run_pipeline(datadir, resdir, data, do_L=False)

