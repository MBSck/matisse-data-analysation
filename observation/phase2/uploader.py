import os
import pprint
import p2api

from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

# TODO: Start with automated folder sorting and creation, will make the process
# faster already

def ob_uploader(path: Path, environment: str, username: str,
                password: str, run_id: int) -> int:
    """Creates folders and uploades the obs

    Parameters
    ----------
    path: Path
        The path to the ob-files of a certain run.
    environment: str
        The enviroment to which is uploaded, 'demo' for testing, 'production'
        for paranal and 'production_lasilla' for la silla
    username: str
        The username for P2
    password: str
        The password for P2
    run_id: int
        The id that specifies the run
    """
    # Environment is either 'demo' for testing, 'production' for paranal or 'production_lasilla", see help
    api = p2api.ApiConnection(environment, username, password)

    container_id_dict = {}

    files = glob(os.path.join(path, "*.obx"))

    # Gets the folder names and creates them
    for i in files:
        stem = os.path.basename(i).split(".")[0]
        if "SCI" in stem:
            folder_name = " ".join([j for j in stem.split("_")[1:]])
            folder_id = create_folder(api, folder_name, run_id)

            # Gets the ids' for the '.obx'-files
            container_id_dict[folder_name] = folder_id

    return 0

def create_folder(api, name: str, container_id: int) -> int:
    """Creates a folder in either the run or the specified directory

    Parameters
    ----------
    api
    name: str
        The folder's name
    container_id: int
        The id that specifies the run

    Returns
    -------
    folder_id: int
    """
    folder, folderVersion = api.createFolder(container_id, name)
    folder_id = folder["containerId"]
    print(f"folder: {name} created!")
    return folder_id

def create_ob(api, ob_name: str, folder_id: int) -> Dict:
    """Creates an ob and edits it with the information of '.obx'-files created
    with Jozsef Vaga's software

    Parameters
    ----------
    api
    ob_name: str
        The name of the ob
    folder_id: int
        The id of the folder the ob gets created in
    """
    ob, obVersion = api.createOB(folder_id, ob_name)
    ob_id = ob["obId"]
    print(f"ob: {ob_name} created!")

# api.generateFindingChart(obId)

if __name__ == "__main__":
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/obmaking/obs"
    ob_uploader(path, "demo", "52052", "tutorial", 1538878)

