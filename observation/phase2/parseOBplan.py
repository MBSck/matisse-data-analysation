#!/usr/bin/env python3

import os
import pickle

from typing import Any, Dict, List, Union, Optional
from pathlib import Path

def readout_txt(file):
    """Reads a txt into its individual lines"""
    with open(file, "r+") as f:
        return f.readlines()

def save_night_lst(input_dict: List, output_path: str):
    """Saves the output dict into a pickle file"""
    with open(output_path, "wb") as fp:
        pickle.dump(input_dict, fp)

def get_nights(file: Path, run_identifier: str = "Night",
               default_key: str = "full_night") -> Dict[str, List]:
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
    default_key: str, optional
        The default identifier that is choosen as a key if 'run_identifier' has
        no matches. Defaults to 'full_night'

    Returns
    -------
    night_dict: Dict
        A dict that contains the <default_search_param> as key and a list
        containing the sub lists 'sci_lst', 'cal_lst' and 'tag_lst'
    """
    lines = readout_txt(file)
    run_indices_lst = [i for i, o in  enumerate(lines) if run_identifier in o]
    run_labels = [i.replace('\n', '') for i in lines if run_identifier in i]

    # TODO: Enable this for bigger night plans that also contain the night and
    # the run as an identifier

    # Debug, remove
    print(run_indices_lst)
    print(run_labels)

    # If there are no specified search terms, whole list will be parsed
    if len(run_indices_lst) == 0:
        run_indices_lst, run_labels = [1], [default_key]

    # Only gets the lines of the runs starting with numbers (SCI or CAL)
    runs = [lines[o:] if o == run_indices_lst[~0] else \
                  lines[o:run_indices_lst[i+1]] for i, o in enumerate(run_indices_lst)]
    runs_cal_sci = [[j for j in i if j[0] in ["1", "0"]] for i in runs]

    runs_dict = {}

    # Gets the sci-, cal-, and ln-list for the nights
    for i, o in enumerate(runs_cal_sci):
        sci_lst, cal_lst, tag_lst = [], [], []
        counter = -1
        for j in o:
            j = j.split(" ")
            print(j)
            if "cal" in j[1]:
                temp_cal = (j[1]+j[2]).split("_")
                tag_lst[counter].append(temp_cal[1])
                cal_lst[counter].append(temp_cal[2])
            else:
                counter += 1
                cal_lst.append([])
                tag_lst.append([])
                sci_lst.append(j[1]+" "+j[2])
        runs_dict[run_labels[i]] = [sci_lst, cal_lst, tag_lst]
    print(runs_dict)
    runs_dict = {}
    print(runs_dict)

    return runs_dict


if __name__ == "__main__":
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/P108/march2022/p108_MATISSE_YSO_runs_observing_plan_v0.1.txt"
    print(path)
    nights = get_nights(path)
    # save_night_lst()

