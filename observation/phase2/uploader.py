import os
import sys
import pprint
import p2api

from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import loadobx

"""

"""

def get_corresponding_run(p2, period: str, proposal_tag: str, number: int):
    """Gets the run that corresponds to the period, proposal and the number and
    returns its runId

    Parameters
    ----------
    p2
    period: str
        The period the run is part of
    proposal_tag: str
        The proposal the run is part of, specifically the tag that is in the
        run's name
    number: int
        The number of the run

    Returns
    -------
    run_id: str
        The run's Id that can be used to access and modify it with the p2api
    """
    runs = p2.getRuns()
    for i in runs[0]:
        run_period, run_proposal, run_number = i["progId"].split(".")
        run_number = int(run_number)

        if (run_period == period) and (run_proposal == proposal_tag) \
           and (run_number == number):
            return i["containerId"]

    print("No matching runs found!")
    return None

def create_folder(p2, name: str, container_id: int) -> int:
    """Creates a folder in either the run or the specified directory

    Parameters
    ----------
    p2
        The ApiConnection to ESO
    name: str
        The folder's name
    container_id: int
        The id that specifies the run

    Returns
    -------
    folder_id: int
    """
    folder, folderVersion = p2.createFolder(container_id, name)
    folder_id = folder["containerId"]
    print(f"folder: {name} created!")
    return folder_id

def get_existing_folders(p2, run_id: str):
    """Fetches the folders' names and ids in the run from the p2 api"""
    folders_dict = {}
    ...

def get_subfolders(path: Path):
    """Fetches the subfolders of a directory"""
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def check4folder(p2, container_id: int) -> bool:
    """Checks if the a container with this id exists, if not then returns
    None"""
    if container_id is None:
        return False

    try:
        if p2.getContainer(o):
            return True
    except p2api.p2api.P2Error:
        return False

def generate_finding_chart_verify():
    ...

def make_folders4OBs(p2, files: List[Path], container_id: int):
    """Makes the respective folders for a list of (.obx)-files

    Parameters
    ----------
    p2: p2api
    files: List[Path]
    container_id: int

    Returns
    -------
    Dict
        A dictionary with the names of the folders and their container_ids
    """
    container_id_dict = {}
    for i in files:
        stem = os.path.basename(i).split(".")[0]
        if "SCI" in stem:
            sci_name = " ".join([j for j in stem.split("_")[1:]])
            folder_id = create_folder(p2, sci_name, container_id)
            loadobx.loadob(p2, i, folder_id)

            for j in files:
                stem_search = os.path.basename(j).split(".")[0]
                if "CAL" in stem_search:
                    sci4cal_name = " ".join(stem_search.split("_")[2:-1])
                    if sci_name == sci4cal_name:
                        loadobx.loadob(p2, j, folder_id)


def ob_uploader(path: Path, server: str, run_data: List,
                username: str, password: Optional[str] = None) -> int:
    """Creates folders and uploades the obs

    Parameters
    ----------
    path: Path
        The path to the ob-files of a certain run.
    server: str
        The enviroment to which the (.obx)-file is uploaded, 'demo' for testing,
        'production' for paranal and 'production_lasilla' for la silla
    run_data: List
        The data that is used to get the runId. The input needs to be in the
        form of a list [run_period: str, run_proposal: str, run_number: int].
        If the list does not contain the run_number, the script looks through
        the folders and fetches it automatically (e.g., run1, ...)
    username: str
        The username for P2
    password: str, optional
        The password for P2, if not given then prompt asking for it is called
    """
    # TODO: Make manual entry for run data possible (full_night), maybe ask for
    # prompt for run number and night name
    p2 = loadobx.login(username, password, server)
    top_dir = glob(os.path.join(path, "*"))

    # TODO: Implement if there is also a standalone setting, that the same
    # nights are used for the standalone as well
    if len(run_data) == 3:
        run_id = get_corresponding_run(p2, *run_data)

    for i in top_dir:
        night_folder_id_dict, main_folder_id_dict = {}, {}
        runs = glob(os.path.join(i, "*"))
        for j in runs:
            if len(run_data) < 3:
                run_number = int(''.join([i for i in os.path.basename(j)\
                                          if i.isdigit()]))
                run_id = get_corresponding_run(p2, *run_data, run_number)

            nights = glob(os.path.join(j, "*"))
            for k in nights:
                night_name = os.path.basename(k)

                if night_name not in night_folder_id_dict:
                    night_folder_id_dict[night_name] = create_folder(p2, night_name, run_id)
                    main_folder_id_dict[night_name] = create_folder(p2, "main_targets",
                                               night_folder_id_dict[night_name])
                    backup_folder = create_folder(p2, "backup_targets",
                                                  night_folder_id_dict[night_name])

                mode_folder_id = create_folder(p2, os.path.basename(i),
                                               main_folder_id_dict[night_name])

                obx_files = glob(os.path.join(k, "*.obx"))
                make_folders4OBs(p2, obx_files, mode_folder_id)
            break

    return 0

if __name__ == "__main__":
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/phase2/obs"
    run_data = ["109", "2313"]
    ob_uploader(path, "production", run_data, "MbS", "QekutafAmeNGNVZ")

