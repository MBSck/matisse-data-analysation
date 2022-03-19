#!/usr/bin/env python3

import os
import time
import subprocess

PATH2SCRIPT = "/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/oca_pipeline/tools/automaticPipeline.py"

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
        path_lst = ["coherent" if i else "incoherent", "lband" if do_l else "nband" ]
        path = "/".join(path_lst)
        script_params = set_script_arguments(rawdir, calibdir, resdir,
                                             do_l, do_flux=i)
        subdir = os.path.join(resdir, path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        process_name = "python " + PATH2SCRIPT + " " + script_params
        process = subprocess.run(process_name, shell=True)

        try:
            os.system(f"mv -f {os.path.join(resdir, 'Iter1/*.rb')} {subdir}")
        except Exception as e:
            print(e)

        toc = time.perf_counter()
        print(f"Executed the {path_lst[0]} {path_lst[1]} reduction in {tic-tic}"
              " seconds")

    return 0

if __name__ == "__main__":
    rawdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/RAW/20190514"
    calibdir = rawdir
    resdir = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/test"

    reduction_pipeline(rawdir, calibdir, resdir, do_l=True)
