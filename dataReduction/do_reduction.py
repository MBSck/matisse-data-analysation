#!/usr/bin/env python3

import os
import time
import subprocess

from typing import Any, Dict, List, Union, Optional

from mat_tools import mat_autoPipeline as mp

# TODO: Look up high spectral binning and make savefile somehow show all
# High spectral binning is 7, 49

def set_script_arguments(do_flux: bool, array: str,
                         spectral_binning: List = [5, 7]) -> str:
    """Sets the arguments that are then passed to the 'mat_autoPipeline.py'
    script"""
    binning_L, binning_N = spectral_binning

    tel = 3 if array == "AT" else 0
    flux = "corrFlux=TRUE/useOpdMod=TRUE/coherentAlgo=2/" if do_flux else ""

    paramL_lst = f"/spectralBinning={binning_L}/{flux}compensate='pb,rb,nl,if,bp,od'"
    paramN_lst = f"/replaceTel={tel}/{flux}spectralBinning={binning_N}"

    return (paramL_lst, paramN_lst)

def single_reduction(rawdir: str, calibdir: str, resdir: str,
                     array: str, mode: bool, band: bool):
    """Reduces either the lband or the nband data for 'coherent' and
    'incoherent'"""
    start_time = time.time()

    path_lst = ["coherent" if mode else "incoherent", "lband" if band else "nband" ]
    path = "/".join(path_lst)
    subdir = os.path.join(resdir, path)
    paramL, paramN = set_script_arguments(mode, array)
    skipL, skipN = int(not band), int(band)

    # Removes the old '.sof'-files
    try:
        os.system(f"rm {os.path.join(resdir, 'Iter1/*.sof*')}")
        os.system(f"rm -r {os.path.join(resdir, 'Iter1/*.rb')}")
        os.system(f"rm -r {os.path.join(subdir, '*.rb')}")
        print("Old files deleted!")
    except Exception as e:
        print("Removing of '.sof'- and '.rb'-files to {subdir} failed!")
        print(e)

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    mp.mat_autoPipeline(dirRaw=rawdir, dirResult=resdir,
                              dirCalib=calibdir,
                              nbCore=6, resol='',
                              paramL=paramL, paramN=paramN,
                              overwrite=0, maxIter=1,
                              skipL=skipL, skipN=skipN)
    try:
        os.system(f"mv -f {os.path.join(resdir, 'Iter1/*.rb')} {subdir}")
    except Exception as e:
        print("Moving of files to {subdir} failed!")
        print(e)

    # Takes the time at end of execution
    end_time = time.time()
    print(f"Executed the {path_lst[0]} {path_lst[1]} reduction in"
          f" {start_time-end_time} seconds")

def reduction_pipeline(rawdir: str, calibdir: str, resdir: str,
                       array: str, both: bool = False,
                       lband: bool = False) -> int:
    """Runs the pipeline for coherent and incoherent reduction."""
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    overall_start_time = time.time()

    if both:
        for i in [True, False]:
            for j in [True, False]:
                single_reduction(rawdir, calibdir, resdir, array,\
                                 mode=i, band=j)
    else:
        for i in [True, False]:
            if lband:
                single_reduction(rawdir, calibdir, resdir, array,\
                                 mode=i, band=True)
            else:
                single_reduction(rawdir, calibdir, resdir, array,\
                                 mode=i, band=False)

    overall_end_time = time.time()
    print(f"Executed the overall reduction in {overall_end_time-overall_start_time}"
          " seconds")
    return 0

if __name__ == "__main__":
    rawdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/RAW/20190514"
    calibdir = rawdir
    resdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514"

    reduction_pipeline(rawdir, calibdir, resdir, "UT", lband=True)
