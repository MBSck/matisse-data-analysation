#!/usr/bin/env python3

import os
import time
import subprocess

from typing import Any, Dict, List, Union, Optional

from mat_tools import mat_autoPipeline as mp

# TODO: Look up high spectral binning and make savefile somehow show all

# important values
PATH2SCRIPT = "/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/oca_pipeline/tools/automaticPipeline.py"

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

def reduction_pipeline(rawdir: str, calibdir: str, resdir: str,
                       array: str) -> int:
    """Runs the pipeline for coherent and incoherent reduction."""
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    overall_start_time = time.time()

    for i in [True, False]:
        for j in [True, False]:
            # Takes the time at the start of execution
            start_time = time.time()

            path_lst = ["coherent" if j else "incoherent", "lband" if i else "nband" ]
            path = "/".join(path_lst)
            paramL, paramN = set_script_arguments(j, array)
            skipL, skipN = int(not i), int(i)

            subdir = os.path.join(resdir, path)
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
                print(e)

            # Takes the time at end of execution
            end_time = time.time()
            print(f"Executed the {path_lst[0]} {path_lst[1]} reduction in"
                   " {start_time-end_time} seconds")

    overall_end_time = time.time()
    print(f"Executed the overall reduction in {overall_start_time-overall_end_time}"
          " seconds")

    return 0

if __name__ == "__main__":
    rawdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/RAW/20190514"
    calibdir = rawdir
    resdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514"

    reduction_pipeline(rawdir, calibdir, resdir, "UT")

