#!/usr/bin/env python3

"""Automated OB Creation

This script creates the OBs from either a manual input list a manual input
dictionary, or a (.yaml)-file and makes folders corresponding to the structure
of the MATISSE observations on the P2

DISCLAIMER: Standalone only works for the UTs at the moment and for ATs it
            shows keyerror

This file can also be imported as a module and contains the following functions:

    * load_yaml - Loads a (.yaml)-file into a dictionary (compatible/superset
    with/of (.json)-format/file)
    * get_night_name_and_date - Quality of life function that gets the night's
    name and date and names
    * get_array - Automatically gets the array config if possible and if not
    prompts the user to input it
    * make_sci_obs - Makes the SCI-OBs
    * make_cal_obs - Makes the CAL-OBs
    * read_dict_into_OBs - Reads either a (.yaml)-file or a Dict into a format
    suitable for OB creation
    * ob_creation - The main loop for OB creation

Example of usage:

    >>> from automatedOBcreation import ob_pipeline

    Script needs either manual_lst, path2file (to the parsed night plan) in
    (.yaml)-file format or run_data (the parsed night plan without it being
    saved)

    >>> path2file = "night_plan.yaml"

    or

    >>> from parseOBplan import parse_night_plan
    >>> run_data = parse_night_plan(...)

    Calibration lists accept sublists (with more than one item)
    but also single items, same for tag_lst

    >>> sci_lst = ["AS 209", "VV Ser"]
    >>> cal_lst = [["HD 142567", "HD 467893"], ["HD 142567"]]
    >>> tag_lst = [["LN", "N"], "L"]
    >>> manual_lst = [sci_lst, cal_lst, tag_lst]

    Specifies the paths, where the (.obx)-files are saved to, name of run can
    be changed to actual one

    >>> outdir = "..."

    Specifies the 'res_dict'. Can be left empty. Changes only L-band res at
    the moment.
    Resolutions are 'LOW', 'MED', 'HIGH'

    >>> res_dict = {"AS 209": "MED"}

    The main OB creation
    Either with 'path2file' (saved (.yaml)-file)

    >>> ob_creation(outpath, path2file=path2file, res_dict=res_dict, mode="gr")

    Or with 'run_data' (parsed (.yaml)-dictionary)

    >>> ob_creation(outpath, run_data=run_data, res_dict=res_dict, mode="gr")

    Or with 'manual_lst' (List input by hand)

    >>> ob_creation(outpath, manual_lst=manual_lst, res_dict=res_dict, mode="gr")
    # ... Making OBs for run 1, 109.2313.001 = 0109.C-0413(A), ATs small array
    # ... Creating folder: night 1 - May 25, and filling it with OBs
    # ... SCI HD 142527         15:56:41.888  -42:19:23.248   4.8      9.8   5.0   8.3
"""

__author__ = "Marten Scheuck"
__date__   = "2022-05"

import os
import yaml
import time
import logging

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union, Optional

import MATISSE_create_OB_2 as ob

# TODO: Make this work for N-band as well
# TODO: Check how to act if H_mag error occurs

# Logging configuration

if os.path.exists("automatedOBcreation.log"):
    os.system("rm -rf autmatedOBcreation.log")
logging.basicConfig(filename='automatedOBcreation.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.INFO)

# Dicts for the template and resolution configuration

UT_DICT_STANDALONE = {"ACQ": ob.acq_tpl,
                      "LOW": {"TEMP": [ob.obs_tpl], "DIT": [0.111],
                              "RES": ["L-LR_N-LR"]}}

AT_DICT_GRA4MAT = {"ACQ": ob.acq_ft_tpl,
                   "LOW": {"TEMP": [ob.obs_ft_tpl], "DIT": [1.], "RES":
                           ["L-LR_N-LR"]},
                   "MED": {"TEMP": [ob.obs_ft_tpl],
                           "DIT": [1.3], "RES": ["L-MR_N-LR"]},
                   "HIGH": {"TEMP": [ob.obs_ft_tpl],
                           "DIT": [3.], "RES": ["L-HR_N-LR"]}}

UT_DICT_GRA4MAT = {"ACQ": ob.acq_ft_tpl,
                   "LOW": {"TEMP": [ob.obs_ft_vis_tpl], "DIT": [0.111], "RES":
                           ["L-LR_N-LR"]},
                   "MED": {"TEMP": [ob.obs_ft_coh_tpl, ob.obs_ft_vis_tpl],
                           "DIT": [1.3, 0.111], "RES": ["L-MR_N-LR"]},
                   "HIGH": {"TEMP": [ob.obs_ft_coh_tpl, ob.obs_ft_vis_tpl],
                           "DIT": [3., 0.111], "RES": ["L-HR_N-LR"]}}

TEMPLATE_RES_DICT = {"standalone": {"UTs": UT_DICT_STANDALONE},
                     "GRA4MAT_ft_vis": {"UTs": UT_DICT_GRA4MAT,
                                        "ATs": AT_DICT_GRA4MAT}}
def load_yaml(file_path):
    """Loads a '.yaml'-file into a dictionary"""
    with open(file_path, "r") as fy:
        return yaml.safe_load(fy)

def get_night_name_and_date(night_key: str) -> str:
    """Automatically gets the night's date if it is included in the
    dictionary

    Parameters
    ----------
    night_key: str
        The dictionaries key that describes a night

    Returns
    -------
    night_str: str
        If night date in night then of the format <night>_<night_date> if not
        then <night>
    """
    night = night_key.split(":")[0].strip()
    date = night_key.split(":")[1].split(",")[0].strip()

    if "night" in night.split(",")[0]:
        night = night_key.split(",")[0].strip()
        date = night_key.split(",")[1].split(",")[0].strip()

    return night + " - " + date if date else night


def get_array(run_name: Optional[str] = None) -> str:
    """Fetches the array configuration from the name of the run. And if no run
    name is specified or no match can be found prompts the user for the
    configuration

    Parameters
    ----------
    run_name: str, optional
        The name of the run

    Returns
    -------
    array_config: str
    """
    at_config = ["small", "medium", "large"]
    tel_config = ["UTs", *at_config]

    if run_name:
        if "UTs" in run_name:
            return "UTs"
        elif ("ATs" in run_name) or (at_config in run_name):
            if "small" in run_name:
                return "small"
            elif "medium" in run_name:
                return "medium"
            elif "large" in run_name:
                return "large"
        else:
            user_inp = int(input("No configuration can be found, please input"\
                             " ('UTs': 1; 'small': 2, 'medium': 3, 'large: 4): "))

            user_inp -= 1
            return tel_config[user_inp]
    else:
        user_inp = int(input("No configuration can be found, please input"\
                         " ('UTs': 1; 'small': 2, 'medium': 3, 'large: 4): "))

        user_inp -= 1
        return tel_config[user_inp]

def make_sci_obs(sci_lst: List, array_config: str, mode: str,
                 outdir: str, res_dict: Dict, standard_res: List) -> None:
    """Gets the inputs from a list and calls the 'mat_gen_ob' for every list element

    Parameters
    ----------
    sci_lst: list
        Contains the science objects
    array_config: str
        The array configuration ('small', 'medium', 'large') or 'UTs'
    mode: str
        The mode of operation of MATISSE
    outdir: str
        The output directory, where the '.obx'-files will be created in
    standard_res: List
        The default spectral resolutions for L- and N-band. Set to low for both
        as a default
    """
    array_key = "UTs" if array_config == "UTs" else "ATs"
    template = TEMPLATE_RES_DICT[mode][array_key]
    ACQ = template["ACQ"]

    if not standard_res:
        standard_res = "LOW" if array_config == "UTs" else "MED"

    for i, o in enumerate(sci_lst):
        try:
            if res_dict and (o in res_dict):
                temp = SimpleNamespace(**template[res_dict[o]])
            else:
                temp = SimpleNamespace(**template[standard_res])

            ob.mat_gen_ob(o, array_config, 'SCI', outdir=outdir,\
                          spectral_setups=temp.RES, obs_tpls=temp.TEMP,\
                          acq_tpl=ACQ, DITs=temp.DIT)
            logging.info(f"Created OB SCI-{o}-#{i}")

        except Exception as e:
            logging.error("Skipped - OB", exc_info=True)
            print("ERROR: Skipped OB - Check (.log)-file")

def make_cal_obs(cal_lst: List, sci_lst: List, tag_lst: List,
                 array_config: str, mode: str, outdir: str,
                 res_dict: Dict, standard_res: List) -> None:
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
    mode: str
        The mode of operation of MATISSE
    outdir: str
        The output directory, where the '.obx'-files will be created in
    standard_res: List, optional
        The default spectral resolutions for L- and N-band. Set to low for both
        as a default
    """
    array_key = "UTs" if array_config == "UTs" else "ATs"
    template = TEMPLATE_RES_DICT[mode][array_key]
    ACQ = template["ACQ"]

    if not standard_res:
        standard_res = "LOW" if array_config == "UTs" else "MED"

    # NOTE: Iterates through the calibration list
    for i, o in enumerate(cal_lst):
        try:
            if res_dict and (sci_lst[i] in res_dict):
                temp = SimpleNamespace(**template[res_dict[sci_lst[i]]])
            else:
                temp = SimpleNamespace(**template[standard_res])

            # NOTE: Checks if list item is itself a list
            if isinstance(o, list):
                for j, l in enumerate(o):
                    ob.mat_gen_ob(l, array_config, 'CAL', outdir=outdir,\
                                  spectral_setups=temp.RES, obs_tpls=temp.TEMP,\
                                  acq_tpl=ACQ, sci_name=sci_lst[i], \
                                  tag=tag_lst[i][j], DITs=temp.DIT)
            else:
                ob.mat_gen_ob(o, array_config, 'CAL', outdir=outdir,\
                              spectral_setups=temp.RES,
                              obs_tpls=temp.TEMP,\
                              acq_tpl=ACQ, sci_name=sci_lst[i],\
                              tag=tag_lst[i], DITs=temp.DIT)

            logging.info(f"Created OB CAL-#{i}")

        except Exception as e:
            logging.error("Skipped - OB", exc_info=True)
            print("ERROR: Skipped OB - Check (.log)-file")

def read_dict_into_OBs(path2file: Path, outpath: Path, mode: str,
                       run_data: Optional[Dict] = None,
                       res_dict: Optional[Dict] = None,
                       standard_res: Optional[List] = None) -> None:
    """This reads either the (.yaml)-file into a format suitable for the Jozsef
    Varga's OB creation code or reads out the run dict if 'run_data' is given,
    and subsequently makes the OBs.

    Also automatically gets the array_config from the run name and if not
    possible then prompts the user to input it

    Parameters
    ----------
    path2file: Path
        The night plan (.yaml)-file
    outpath: Path
        The output path
    mode: str
        The mode of operation of MATISSE
    res_dict: Dict, optional
        A dict with entries corresponding to non low-resolution
    standard_res: List, optional
        The default spectral resolutions for L- and N-band. By default it is set
        to medium for L- and low for N-band
    """
    if run_data:
        run_dict = run_data
    else:
        run_dict = load_yaml(path2file)

    for i, o in run_dict.items():
        print(f"Making OBs for {i}")
        logging.info(f"OBs for {i}")
        run_name = i.split(",")[0].strip()

        array_config = get_array(i)
        for j, l in o.items():
            night_name = get_night_name_and_date(j)
            temp_path = os.path.join(outpath, run_name, night_name)

            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

            print(f"Creating folder: {night_name}, and filling it with OBs")
            logging.info(f"Creating folder: {night_name}, and filling it with OBs")

            # NOTE: To not get a timeout from the databases
            time.sleep(1)

            night = SimpleNamespace(**l)
            make_sci_obs(night.SCI, array_config, mode, temp_path, res_dict,
                         standard_res)
            make_cal_obs(night.CAL, night.SCI, night.TAG, array_config, mode,
                         temp_path, res_dict, standard_res)

def ob_creation(outpath: str, path2file: Optional[Path] = None,
                run_data: Optional[Dict] = None, manual_lst: Optional[List] = None,
                res_dict: Optional[Dict] = None,
                standard_res: Optional[List] = None,
                obs_templates: Optional[List] = None,
                acq_template = None, mode: str = "st") -> int:
    """Gets all functionality and automatically creates the OBs.

    Parameters
    ----------
    array_config: str
        The array configuration
    path2file: str, optional
        The path to the dictionary file, if exists
    run_data; Dict, optional
        The parsed data of the night plan
    manual_lst: List, optional
        The manual list input
    res_dict: Dict, optional
        A dict with entries corresponding to non low-resolution
    standard_res: List, optional
        The default spectral resolutions for L- and N-band. By default it is set
        to medium for L- and low for N-band
    mode: str
        The mode MATISSE is operated in and for which the OBs are created.
        Either 'st' for standalone, 'gr' for GRA4MAT_ft_vis or 'both',
        if OBs for both are to be created

    Returns
    -------
    Execution_code
        '1' for SUCCESS '-1' for ERROR
    """
    mode_lst = ["standalone", "GRA4MAT_ft_vis"] if mode == "both" else \
            (["standalone"] if mode == "st" else ["GRA4MAT_ft_vis"])

    for i in mode_lst:
        if manual_lst:
            sci_lst, cal_lst, tag_lst = manual_lst
            outpath = os.path.join(outpath, "manualOBs")

            if not os.path.exists(outpath):
                os.makedirs(outpath)

            array_config = get_array()

            make_sci_obs(sci_lst, array_config, i, outpath, res_dict,
                         standard_res)
            make_cal_obs(cal_lst, sci_lst, tag_lst, array_config, i,\
                         outpath, res_dict, standard_res)

        elif path2file or run_data:
            outpath = os.path.join(outpath, i)
            read_dict_into_OBs(path2file, outpath, i, run_data,
                               res_dict, standard_res)

        elif IOError:
            raise IOError("Neither '.yaml'-file nor input list found or input"
                          " dict found!", -1)

    return 0


if __name__ == "__main__":
    path2file = "night_plan.yaml"
    outdir = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/phase2/obs/"

    sci_lst = [""]
    cal_lst = [""]
    tag_lst = [""]
    manual_lst = [sci_lst, cal_lst, tag_lst]

    res_dict = {}

    ob_creation(outdir, path2file=path2file, res_dict=res_dict, mode="gr")

