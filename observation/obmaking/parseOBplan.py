#!/usr/bin/env python3

import os
import pickle

from typing import Any, Dict, List, Union, Optional

def readout_txt(file):
    with open(file, "r+") as f:
        return f.readlines()

def get_nights(lines: List, search_term: str = "Night",
               default_name: str = "Full_Night") -> Dict:
    """
    Parameters
    ----------
    lines: List
        The individual lines of the file
    search_term: str, optional
        Set to default as 'Night'. Sets the search term
    default_name: str, optional
        The default output folder that the OBs are put into

    Returns
    -------
    night_dict: Dict
        A dict that contains the <default_search_param> as key and a list
        containing the 'sci_lst', 'cal_lst' and 'tag_lst'
    """
    # Sorts the nights after indices and makes the labels
    night_indices_lst = [i for i, o in  enumerate(lines) if "Night" in o]
    night_labels = [i for i in lines if "Night" in i]

    # If there are no specified search terms it will get whole list
    if not night_indices_lst:
        night_indices_lst.append(1)
        night_labels.append(default_name)

    nights = [[] for _ in range(len(night_indices_lst))]


    # Gets the lines for every night
    for i, o in enumerate(night_indices_lst):
        if i != len(night_indices_lst)-1:
            temp_lines = lines[night_indices_lst[i]:night_indices_lst[i+1]]
        else:
            temp_lines = lines[o:]
        night_lines = [i for i in temp_lines if i[0] == "1" or i[0] == "0"]
        nights[i] = night_lines

    # Night lists
    night_dict = {}

    # Gets the sci-, cal-, and ln-list for the nights
    for i, o in enumerate(nights):
        sci_lst, cal_lst, tag_lst = [], [], []
        counter = -1
        for j in o:
            j = j.split(" ")
            if "cal" in j[1]:
                temp_cal = (j[1]+j[2]).split("_")
                tag_lst[counter].append(temp_cal[1])
                cal_lst[counter].append(temp_cal[2])
            else:
                counter += 1
                cal_lst.append([])
                tag_lst.append([])
                sci_lst.append(j[1]+" "+j[2])
        night_dict[night_labels[i]] = [sci_lst, cal_lst, tag_lst]

    return night_dict


def save_night_lst(input_dict: List, output_path: str):
    with open(output_path, "wb") as fp:
        pickle.dump(input_dict, fp)


if __name__ == "__main__":
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/P108/march2022"
    file = "p108_MATISSE_YSO_runs_observing_plan_v0.3.txt"
    path2file = os.path.join(path, file)

    readout = readout_txt(path2file)

    nights = get_nights(readout)
    save_night_lst(nights, os.path.join(path, "nights_OB.txt"))
    # save_night_lst()

