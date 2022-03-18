#!/usr/bin/env python3

import os
import time
import subprocess

PATH2SCRIPT = 

def set_script_arguments(rawdir: str, calibdir: str, resdir: str,
                         do_l: bool, do_flux: bool) -> str:
    """Sets the arguments that are then passed to the 'automaticPipline.py'
    script"""
    directories = f" --dirRaw={rawdir} --dirCalib={calibdir} --dirResult={resdir} "
    general_params = "--nbCore=10 --overwrite=TRUE --maxIter=1 "

    if do_flux:
        corr_flux = "TRUE"
    else:
        corr_flux = "FALSE"

    param_l_band = f"--paramL=/corrFlux={corr_flux}/coherentAlgo=2/compensate=[pb,rb,nl,if,bp,od]/cumulBlock=TRUE/spectralBinning=11/ "

    if do_l:
        additional_params = "--skipN"
    else:
        additional_params = "--skipL"

    return directories+general_params+param_l_band+additional_params

def reduction_pipeline(rawdir: str, calibdir: str, resdir: str, do_l: bool) -> int:
    """Runs the pipeline for coherent and incoherent reduction."""
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    for i in [True, False]:
        tic = time.perf_counter()
        path = "/".join(path_lst := ["coherent" if i else "incoherent", "lband" if do_l else "nband" ])
        script_params = set_script_arguments(rawdir, calibdir, resdir,
                                             do_l, do_flux=i)
        if not os.path.exists(subdir := os.path.join(resdir, path)):
            os.makedirs(subdir)

        subprocess.call(["python", PATH2SCRIPT, script_params], stdout=PIPE, stderr=PIPE)

        try:
            os.system(f"mv -f {os.path.join(resdir, 'Iter1/*.rb')} {subdir}")
        except Exception as e:
            print(e)

        toc = time.perf_counter()
        print(f"Executed the {path_lst[0]} {path_lst[1]} reduction in {tic-tic}"
              " seconds")

    return 0

if __name__ == "__main__":
    rawdir = "Hello"
    calibdir = "Peter"
    resdir = "Pedigrew"

    reduction_pipeline(rawdir, calibdir, resdir, do_l=True)
