__author__ = "Marten Scheuck"

import os
import pickle
import time

from typing import Any, Dict, List, Union, Optional

import MATISSE_create_OB_2 as ob
import parseOBplan

# TODO: Implement this for MATISSE standalone as well
# TODO: Fix the fact that happens when a calibrator is first before the SCI
# TODO: Implement various setups, long lists for templates etc.
# TODO: Make this work for n-band as well
# TODO: High res cal file is erroneous
# TODO: Make standard setting for resolution possible
# TODO: Make setting both resolutions possible
# TODO: Set dit to 0.111 for standalone matisse
# TODO: Reset all changes after the OB creation

def load_dict(file_path):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)

def set_res(standard_res: List):
    """placeholder function until setting both res is understood with josef's
    code"""
    # TODO: Remove this and make it work with get_res_dit function for all of
    # the code
    dit = 1.

    if "MR" in standard_res:
        dit = 1.3
    if "HR" in standard_res:
        dit = 3.

    return f"L-{standard_res[0]}_N-{standard_res[1]}", dit

def get_res_dit(resolution: str):
    """Gets the dit for the resolution input

    Parameters
    ----------
    resolution: str
        In either 'MR' or 'HR'

    Returns
    -------
    resolution: str
        In the right input form for the 'make_cal_obs'
    dit: float
        The corresponding dit to the resolution
    """
    if "MR" in resolution:
        return "L-MR_N-LR", 1.3
    if "HR" in resolution:
        return "L-HR_N-LR", 3.
    return  "L-LR_N-LR", 1.

def make_sci_obs(sci_lst: List, array_config: str,
                 outdir: str, res_dict: Dict, standard_res: List,
                 obs_templates: List, acq_template) -> None:
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
    for i in sci_lst:
        if i in res_dict:
            resolution, dit = get_res_dit(res_dict[i])
        else:
            resolution, dit = set_res(standard_res)
        dit = [0.111, 1.3]

        ob.mat_gen_ob(i, array_config, 'SCI', outdir=outdir,\
                      spectral_setups=[resolution], obs_tpls=obs_templates,\
                      acq_tpl=acq_template, DITs=dit)

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
        # Iterates through the calibration list
        for i, o in enumerate(cal_lst):
            # Sets the resolution and the dit
            if sci_lst[i] in res_dict:
                resolution, dit = get_res_dit(res_dict[sci_lst[i]])
            else:
                resolution, dit = set_res(standard_res)

            dit = [0.111, 1.3]

            # Checks if list item is itself a list
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

def ob_pipeline(array: str, outpath: str, path2file: str = None,
                manual_lst: List = None, res_dict: Dict = None,
                standard_res: List = ["LR", "LR"],
                obs_templates: List = [ob.obs_ft_tpl],
                acq_template = ob.acq_ft_tpl) -> int:
    """Gets all functionality and automatically creates the OBs

    Parameters
    ----------
    array: str
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
    """
    if manual_lst:
        sci_lst, cal_lst, tag_lst = manual_lst
        make_sci_obs(sci_lst, array, outpath, res_dict, standard_res,\
                     obs_templates, acq_template)
        make_cal_obs(cal_lst, sci_lst, tag_lst, array,\
                     outpath, res_dict, standard_res,\
                     obs_templates, acq_template)

    if path2file:
        # Load dict
        nights_dict = load_dict(path2file)

        # Make calibs for sci-file
        for i, o in nights_dict.items():
            temp_path = os.path.join(outpath, i.strip('\n'))

            if not os.path.exists(temp_path):
                os.mkdir(temp_path)

            time.sleep(1)

            sci_lst, cal_lst, tag_lst = o
            make_sci_obs(sci_lst, array, temp_path, res_dict, standard_res,
                         obs_templates, acq_template)
            make_cal_obs(cal_lst, sci_lst, tag_lst, array,\
                         temp_path, res_dict, standard_res,\
                         obs_templates, acq_template)

    return 0


if __name__ == "__main__":
    # Get and parses the night plans Roy's script creates
    try:
        path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/P108/march2022"
        file = "p108_MATISSE_YSO_runs_observing_plan_v0.8.txt"
        path2file = os.path.join(path, file)
        readout = parseOBplan.readout_txt(path2file)
        nights = parseOBplan.get_nights(readout)
        path2file = os.path.join(path, "nights_OB.txt")
        parseOBplan.save_night_lst(input_dict=nights, output_path=path2file)
    except:
        # If non valid path is given then default to empty string
        print("No input file for parsing found or not readable!")
        path2file= ""

    # Specifies the paths, where the '.obx'-files are saved to, name of run can
    # be changed to actual one
    outdir = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/obmaking/obs/"
    outpath = os.path.join(outdir, "run11")

    # Makes the safe path if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Manule use of the wrapper
    # Example of usage, calibration lists accept sublists, more than one
    # calibrator, but also single items, same for tag_lst
    # sci_lst = ["AS 209", "VV Ser"]
    # cal_lst = [["HD 142567", "HD 467893"], "]
    # tag_lst = [["LN", "N"], "L"]

    sci_lst = ["AS 209"]
    cal_lst = []
    tag_lst = []

    # Specifies the res_dict, in the format. Can be left empty.
    # Example of usage, at the moment only changes L-band resolution
    # res_dict = {"AS 209": "MR"}
    res_dict = {}

    # Pipeline for ob creation, parseOB is commented out, templates can be
    # changed
    # Templates that can be used, for acq. only one at a time, for obs multiple
    # in a list
    # ob.acq_ft_tpl, ob.acq_tpl, ob.obs_tpl, ob.obs_ft_tpl, ob.obs_ft_coh_tpl,
    # ob.obs_ft_vis_tpl
    ob_pipeline("UTs", outpath, path2file,  manual_lst=[sci_lst, cal_lst, tag_lst],\
                res_dict=res_dict, standard_res=["LR", "LR"],\
                obs_templates=[ob.obs_ft_vis_tpl], acq_template=ob.acq_ft_tpl)

