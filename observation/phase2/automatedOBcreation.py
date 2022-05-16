#!/usr/bin/env python3

"""Automated OB Creation

...

This file can also be imported as a module and contains the following functions:

    * load_yaml -
    * set_res -
    * make_sci_obs -
    * make_cal_obs -
    * ob_pipeline -

Example of usage:
    >>> from automatedOBcreation import ob_pipeline

    # Script needs either manual_lst or path2file (to the parsed night plan) in
    # (.yaml)-file format

    >>> path2file = "night_plan.yaml"

    # Calibration lists accept sublists (with more than one item)
    # but also single items, same for tag_lst

    >>> sci_lst = ["AS 209", "VV Ser"]
    >>> cal_lst = [["HD 142567", "HD 467893"], "]
    >>> tag_lst = [["LN", "N"], "L"]

    # Specifies the paths, where the '.obx'-files are saved to, name of run can
    # be changed to actual one

    >>> outdir = "..."

    # Specifies the res_dict. Can be left empty. Changes only L-band res atm

    >>> res_dict = {"AS 209": "MR"}

    # Templates that can be used, for acq. only one at a time, for obs multiple
    # For instance ob.acq_ft_tpl, ob.acq_tpl, ob.obs_tpl, ob.obs_ft_tpl,
    # ob.obs_ft_coh_tpl, ob.obs_ft_vis_tpl

    >>> ob_pipeline("UTs", outpath, path2file,
                    manual_lst=[sci_lst, cal_lst, tag_lst],
                    res_dict=res_dict, standard_res=["LR", "LR"],
                    obs_templates=[ob.obs_tpl], acq_template=ob.acq_tpl)
"""

__author__ = "Marten Scheuck"
__date__   = "2022-05-12"

import os
import yaml
import time

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union, Optional

import MATISSE_create_OB_2 as ob

# TODO: Make this work for n-band as well
# TODO: Reset all changes after the OB creation

# Dicts for the template and resolution configuration

UT_DICT_STANDALONE = {"ACQ": ob.acq_tpl,
                      "LOW": {"TEMP": [ob.obs_tpl], "DIT": [0.111],
                              "RES": "L-LR_N-LR"}}

AT_DICT_GRA4MAT = {"ACQ": ob.acq_ft_tpl,
                   "LOW": {"TEMP": [ob.obs_ft_tpl], "DIT": [1.], "RES":
                           "L-LR_N-LR"},
                   "MED": {"TEMP": [ob.obs_ft_tpl],
                           "DIT": [1.3], "RES": "L-MR-LR"},
                   "HIGH": {"TEMP": [ob.obs_ft_tpl],
                           "DIT": [3.], "RES": "L-HR-LR"}}

UT_DICT_GRA4MAT = {"ACQ": ob.acq_ft_tpl,
                   "LOW": {"TEMP": [ob.obs_ft_vis_tpl], "DIT": [0.111], "RES":
                           "L-LR_N-LR"},
                   "MED": {"TEMP": [ob.obs_ft_coh_tpl, ob.obs_ft_vis_tpl],
                           "DIT": [1.3, 0.111], "RES": "L-MR-LR"},
                   "HIGH": {"TEMP": [ob.obs_ft_coh_tpl, ob.obs_ft_vis_tpl],
                           "DIT": [3., 0.111], "RES": "L-HR-LR"}}

TEMPLATE_DICT = {"standalone": {"UTs": UT_DICT_STANDALONE},
                 "GRA4MAT": {"UTs": UT_DICT_GRA4MAT, "ATs": AT_DICT_GRA4MAT}}

def load_yaml(file_path):
    """Loads a '.yaml'-file into a dictionary"""
    with open(file_path, "r") as fy:
        return yaml.safe_load(fy)

def make_sci_obs(sci_lst: List, array_config: str,
                 outdir: str, res_dict: Dict, standard_res: List,
                 obs_templates: List, acq_template, mode: str) -> None:
    """Gets the inputs from a list and calls the 'mat_gen_ob' for every list element

    Parameters
    ----------
    sci_lst: list
        Contains the science objects
    array_config: str
        The array configuration ('small', 'medium', 'large') or 'UTs'
    outdir: str
        The output directory, where the '.obx'-files will be created in
    resolution: str, optional
        The resolution of the OB
    standard_res: List, optional
        The default spectral resolutions for L- and N-band. Set to low for both
        as a default

    Returns
    -------
    None
    """
    try:
        array_key = "UTs" if array_config == "UTs" else "ATs"

        for i in sci_lst:
            if i in res_dict:
                resolution, dit = get_res_dit(res_dict[i])
            else:
                resolution, dit = set_res(standard_res)

            ob.mat_gen_ob(i, array_config, 'SCI', outdir=outdir,\
                          spectral_setups=[resolution], obs_tpls=obs_templates,\
                          acq_tpl=acq_template, DITs=dit)
    except Exception as e:
        print("Skipped OB - Check")
        print(e)

def make_cal_obs(cal_lst: List, sci_lst: List, tag_lst: List,
                 array_config: str, outdir: str,
                 res_dict: Dict, standard_res: List,
                 obs_templates: List, acq_template) -> None:
    """Checks if there are sublists in the calibration list and calls the 'mat_gen_ob' with the right inputs
    to generate the calibration objects.
    The input lists correspond to each other index-wise (e.g., cal_lst[1], sci_lst[1], tag_lst[1]; etc.)

    Parameters
    ----------
    cal_lst: List
        Contains the calibration objects corresponding to the science objects
    sci_lst: List
        Contains the science objects
    tag_lst: List
        Contains the tags (either 'L', 'N', or both) and corresponds to the science objects
    array_config: str
        The array configuration ('small', 'medium', 'large') or 'UTs'
    outdir: str
        The output directory, where the '.obx'-files will be created in
    resolution: str, optional
        The resolution of the OB
    standard_res: List, optional
        The default spectral resolutions for L- and N-band. Set to low for both
        as a default

    Returns
    -------
    None
    """
    try:
        array_key = "UTs" if array_config == "UTs" else "ATs"

        # NOTE: Iterates through the calibration list
        for i, o in enumerate(cal_lst):
            if sci_lst[i] in res_dict:
                resolution, dit = get_res_dit(res_dict[sci_lst[i]])
            else:
                resolution, dit = set_res(standard_res)

            # NOTE: Checks if list item is itself a list
            if isinstance(o, list):
                for j, l in enumerate(o):
                    ob.mat_gen_ob(l, array_config, 'CAL', outdir=outdir,\
                                  spectral_setups=[resolution], obs_tpls=obs_templates,\
                                  acq_tpl=acq_template, sci_name=sci_lst[i], \
                                  tag=tag_lst[i][j], DITs=dit)
            else:
                ob.mat_gen_ob(o, array_config, 'CAL', outdir=outdir,\
                              spectral_setups=[resolution],
                              obs_tpls=obs_templates,\
                              acq_tpl=acq_template, sci_name=sci_lst[i],\
                              tag=tag_lst[i], DITs=dit)

    except Exception as e:
        print("Skipped OB - Check")
        print(e)

def read_yaml_into_OBs(path2file: Path outpath: Path, mode: str) -> None:
    """This reads the (.yaml)-file into a format suitable for the Jozsef
    Varga's OB creation code and subsequently makes the OBs with it

    Parameters
    ----------
    path2file: Path
        The night plan (.yaml)-file
    outpath: Path
        The output path
    mode: str
        The mode of operation of MATISSE
    """
    run_dict = load_yaml(path2file)

    for i, o in run_dict.items():
        for j, l in o.items():
            temp_path = os.path.join(outpath,\
                                     i.split(",")[0].replace(' ', ''),\
                                     j.split(":")[0].replace(' ', ''))

            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

            time.sleep(1)
            night = SimpleNamespace(**l)
            make_sci_obs(night.SCI, array_config, temp_path, res_dict, standard_res,
                         obs_templates, acq_template, mode)
            make_cal_obs(night.CAL, night.SCI, night.TAG, array_config,\
                         temp_path, res_dict, standard_res,\
                         obs_templates, acq_template, mode)

def ob_pipeline(array_config: str, outpath: str, path2file: Optional[Path] = None,
                manual_lst: Optional[List] = None, res_dict: Optional[Dict] = None,
                standard_res: Optional[List] = None,
                obs_templates: Optional[List] = None
                acq_template = None, mode: str = "st") -> int:
    """Gets all functionality and automatically creates the OBs

    Parameters
    ----------
    array_config: str
        The array configuration
    path2file: str, optional
        The path to the dictionary file, if exists
    manual_lst: List, optional
        The manual list input
    res_dict: Dict, optional
        A dict with entries corresponding to non low-resolution
    standard_res: List, optional
        The default spectral resolutions for L- and N-band. Set to low for both
        as a default
    mode: str
        The mode MATISSE is operated in and for which the OBs are created.
        Either 'st' for standalone, 'gr' for GRA4MAT or 'both', if OBs for both
        are to be created

    Returns
    -------
    Execution_code
        '1' for SUCCESS '-1' for ERROR
    """
    if manual_lst[0]:
        sci_lst, cal_lst, tag_lst = manual_lst
        make_sci_obs(sci_lst, array_config, outpath, res_dict, standard_res,\
                     obs_templates, acq_template, mode)
        make_cal_obs(cal_lst, sci_lst, tag_lst, array_config,\
                     outpath, res_dict, standard_res,\
                     obs_templates, acq_template, mode)

    elif path2file:
        mode_lst = ["standalone", "GRA4MAT"] if mode == "both" else \
                (["standalone"] if mode == "st" else ["GRA4MAT"])
        print(mode_lst)

        for i in mode_lst:
            outpath = os.path.join(outpath, i)
            read_yaml_into_OBs(path2file, outpath, i)
    else:
        raise IOError("Neither '.yaml'-file nor input list found!")

    return 0


if __name__ == "__main__":
    path2file = ""

    outdir = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/obmaking/obs/"
    outpath = os.path.join(outdir)

    sci_lst = []
    cal_lst = []
    tag_lst = []

    res_dict = {}

    ob_pipeline("UTs", outpath, path2file,  manual_lst=[sci_lst, cal_lst, tag_lst],\
             res_dict=res_dict}
