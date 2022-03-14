__author__ = "Marten Scheuck"

import os
import pickle
import time

from typing import Any, Dict, List, Union, Optional

import MATISSE_create_OB_2 as ob

# TODO: Make this work for n-band as well
# TODO: High res cal file is erroneous
# TODO: Make standard setting for resolution possible
# TODO: Make setting both resolutions possible

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


# TODO: Add comments and replenish this file
def shell_main():
    """
    This function's sole purpose is to enable the plotter to work in the shell
    """
    try:
        sci_lst, cal_lst, tag_lst, interferometric_array_config, sci_or_cal  = sys.argv[1:5]
    except:
        try:
            sci_lst, interferometric_array_config, sci_or_cal = sys,argv[1:3]
        except:
            print("Usage: python3 myplotter.py /sci_lst/ /cal_lst/ /tar_lst/ /sci/cal/")
            sys.exit(1)

    if sci_or_cal == "sci":
        make_sci_obs(sci_lst, interferometric_array_config, outdir=os.getcwd())

    if sci_or_cal == "cal":
        make_cal_obs(cal_lst, sci_lst, tag_lst, interferometric_array_config, outdir=os.getcwd())

def make_sci_obs(sci_lst: List, interferometric_array_config: str,
                 outdir: str, resolution: str = None,
                 standard_res: List = ["LR", "LR"]) -> None:
    """Gets the inputs from a list and calls the 'mat_gen_ob' for every list element

    Parameters
    ----------
    sci_lst: list
        Contains the science objects
    interferometric_array_config: str
        The array configuration ('small', 'medium', 'large')
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
        for i in sci_lst:
            if i in res_dict:
                resolution, dit = get_res_dit(res_dict[i])
            else:
                resolution, dit = set_res(standard_res)

            ob.mat_gen_ob(i, interferometric_array_config, 'SCI', outdir=outdir,\
                          spectral_setups=[resolution], obs_tpls=[ob.obs_ft_tpl],\
                          acq_tpl=ob.acq_ft_tpl, DITs=[dit])
    except:
        print(f"Skipped OB {i} - Check")

def make_cal_obs(cal_lst: List, sci_lst: List, tag_lst: List,
                 interferometric_array_config: str, outdir: str,
                 resolution: str = None, standard_res: List = ["LR", "LR"]) -> None:
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
    interferometric_array_config: str
        The array configuration ('small', 'medium', 'large')
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

            # Checks if list item is itself a list
            if isinstance(o, list):
                for j, l in enumerate(o):
                    ob.mat_gen_ob(l, interferometric_array_config, 'CAL', outdir=outdir,\
                                  spectral_setups=[resolution], obs_tpls=[ob.obs_ft_tpl],\
                                  acq_tpl=ob.acq_ft_tpl, sci_name=sci_lst[i], \
                                  tag=tag_lst[i][j], DITs=[dit])
            else:
                ob.mat_gen_ob(o, interferometric_array_config, 'CAL', outdir=outdir,\
                              spectral_setups=[resolution], obs_tpls=[ob.obs_ft_tpl],\
                              acq_tpl=ob.acq_ft_tpl, sci_name=sci_lst[i],\
                              tag=tag_lst[i], DITs=[dit])

    except:
        print("Skipped OB - Check")

def ob_pipeline(array: str, outpath: str, path2file: str = None,
                manual_lst: List = None, res_dict: Dict = None,
                standard_res: List = ["LR", "LR"]) -> int:
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
        make_sci_obs(sci_lst, array, outpath, res_dict, standard_res)
        make_cal_obs(cal_lst, sci_lst, tag_lst, array,\
                     outpath, res_dict, standard_res)

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
            make_sci_obs(sci_lst, array, temp_path, res_dict, standard_res)
            make_cal_obs(cal_lst, sci_lst, tag_lst, array,\
                         temp_path, res_dict, standard_res)

    return 0


if __name__ == "__main__":
    # Gets the savepaths
    list_file_path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/P108/march2022/"
    ob_plan_file = "nights_OB.txt"
    path2file = os.path.join(list_file_path, ob_plan_file)
    outdir = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/obmaking/obs/"
    outpath = os.path.join(outdir, "run9")

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Manual lists
    sci_lst = ["R_Mon"]
    cal_lst = ["HD 49161"]
    tag_lst = ["LN"]

    # Specifies the res_dict
    res_dict = {}

    # Pipeline for ob creation
    ob_pipeline("small", outpath, manual_lst=[sci_lst, cal_lst, tag_lst], res_dict=res_dict, standard_res=["MR", "LR"])

    # For shell implementation
    # shell_main()

