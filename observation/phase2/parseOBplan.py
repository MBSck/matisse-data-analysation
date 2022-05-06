#!/usr/bin/env python3


""" Parse OB Plan

This script parses the night plans made with Roy van Boekel's "calibrator_find"
IDL script into a (.yaml)-file that contains the CALs sorted to their
corresponding SCI-targets in a dictionary that first specifies the run, then
the night and then the SCIs, CALs and TAGs (If calibrator is LN/N or L-band).

This tool accepts (.txt)-files.

The script requires that `yaml` be installed within the Python environment this
script is run in.

This file can also be imported as a module and contains the following functions:

    * readout_txt - reads a (.txt)-file into its individual lines returning them
    * save_dictionary - Saves a dictionary as a (.yaml)-file
    * check_lst4elem - Checks a list for an element and returns a bool
    * get_file_section - Gets a section of the (.txt)/lines
    * get_sci_cal_tag_lst - Gets the individual lists of the SCI, CAL and TAG
    * parse_night_plan - The main function of the script. Parses the night plan

"""

import os
import yaml

from pathlib import Path
from typing import Any, Dict, List, Union, Optional


def readout_txt(file):
    """Reads a txt into its individual lines"""
    with open(file, "r+") as f:
        return f.readlines()

def save_dictionary(input_dict: Dict, output_path: Path) -> int:
    """Saves the output dict into a '.yaml'-file. A superset of '.json'-file
    that can easily be read by people as well

    Parameters
    ----------
    input_dict: Dict
        The dictionary to be saved
    output_path: Path
        The path the '.yaml'-file is to be saved to

    Returns
    -------
    int
        ERROR (-1) or SUCCESS (0)
    """
    try:
        with open(output_path, "w") as fy:
            yaml.safe_dump(input_dict, fy)
        return 0
    except Exception as e:
        print(e)
        return -1

def check_lst4elem(input_lst: List, elem: str) -> bool:
    """Checks if the element is in list and returns True or False

    Parameters
    ----------
    input_lst: List
        The input list
    elem: str
        The element being checked for

    Returns
    -------
    bool
        True if element is found, False otherwise
    """
    for i in input_lst:
        if elem in i:
            return True

    return False

def get_file_section(lines: List, identifier: str):
    """Gets the section of a file corresponding to the given identifier and
    returns a dict with the keys being the match to the identifier and the
    values being a subset of the lines list

    Parameters
    ----------
    lines: List
        The lines read from a file
    identifier: str
        The identifier by which they should be split into subsets

    Returns
    --------
    subset: dict
        A dict that contains a subsets of the original lines
    """
    indices_lst, labels = [], []
    default_key = "full_" + identifier
    for i, o in enumerate(lines):
        if identifier in o.lower():
            indices_lst.append(i)
            labels.append(o.replace('\n', ''))

    if len(indices_lst) == 0:
        indices_lst, labels = [0], [default_key]

    sections = [lines[o:] if o == indices_lst[~0] else \
                  lines[o:indices_lst[i+1]] for i, o in enumerate(indices_lst)]

    return {labels: sections for (labels, sections) in zip(labels, sections)}

def get_sci_cal_tag_lst(lines: List):
    """Gets the info for the SCI, CAL and TAGs from the individual lines

    Parameters
    -----------
    lines: List
        The input lines to be parsed

    Returns
    -------
    List:
        The lists in the output format [[SCI], [CAL], [TAG]]
    """
    line_range = [i for i, o in enumerate(lines) if o[0].isdigit()]
    lines = lines[line_range[0]:line_range[~0]+1]

    for i, o in enumerate(lines):
        lines[i] = o.replace('\n', '')

    # TODO: Make a different separating instead of the '' lines

    sci_lst, cal_lst, tag_lst  = [], [[]], [[]]
    counter = 0

    for i, o in enumerate(lines):
        if o == '':
            counter += 1
            cal_lst.append([])
            tag_lst.append([])
        else:
            o = o.split()
            if not o[0][0].isdigit():
                # NOTE: Bodge (crude fix) to skip over irrelevant lines
                # TODO: Make this more robust as well
                continue

            if "cal_" in o[1]:
                temp_cal = o[1].split("_")
                cal_lst[counter].append(temp_cal[2])
                tag_lst[counter].append(temp_cal[1])
            else:
            # FIXME: Make the parse here more robust for numbers (e.g., WA Oph 6)
                if (o[3] != '') and (not o[3].isdigit()):
                    sci_lst.append(o[1]+' '+o[2]+' '+o[3])
                else:
                    sci_lst.append(o[1]+' '+o[2])

    return {"SCI": sci_lst, "CAL": cal_lst, "TAG": tag_lst}

def parse_night_plan(file: Path, run_identifier: Optional[str] = "run",
                     sub_identifier: Optional[str] = "night",
                     save2file: bool = False) -> Dict[str, List]:
    """
    Parses the night plan created with 'calibrator_find.pro' into the
    individual runs as key of a dictionary, specified by the 'run_identifier'.
    If no match is found then it parses the whole night to 'run_identifier's
    or the 'default_key', respectively.

    Parameters
    ----------
    file: Path
        The night plan of the '.txt'-file format to be read and parsed
    run_identifier: str, optional
        Set to default identifier that splits the individual runs into keys of
        the return dict as 'run'
    sub_identifier: str, optional
        Set to default sub identifier that splits the individual runs into the
        individual nights. That is, in keys of the return dict as 'night'
    save2file: bool, optional
        If this is set to true then it saves the dictionary as
        'night_plan.yaml', Default is 'False'

    Returns
    -------
    night_dict: Dict
        A dict that contains the <default_search_param> as key and a list
        containing the sub lists 'sci_lst', 'cal_lst' and 'tag_lst'
    """
    lines = readout_txt(file)
    runs = get_file_section(lines, run_identifier)
    night_plan = {}

    for i, o in runs.items():
        temp_dict = get_file_section(o, sub_identifier)
        nights = {}
        for j, l in temp_dict.items():
            if check_lst4elem(l, "cal_"):
                nights[j] = get_sci_cal_tag_lst(l)

        night_plan[i] = nights

    if save2file:
        save_dictionary(night_plan, "night_plan.yaml")

    return night_plan

if __name__ == "__main__":
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/P109/april2022/"\
            "p109_MATISSE_YSO_runs_observing_plan_v0.3.txt"
    print(run_dict := parse_night_plan(path, save2file=True))

