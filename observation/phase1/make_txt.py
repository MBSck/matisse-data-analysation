import os

from glob import glob
from typing import Any, Dict, List, Union, Optional

def sort_into_list(filepath: str) -> None:
    """Takes an input txt file with comma seperated values and parses it into a
    list with unique entries"""
    run_path = os.path.join("/".join(filepath.split("/")[:-1]), "run_files")

    # Gets file content, seperated by commas
    if not "ft" in filepath:
        with open(filepath, 'r') as fr:
            file_content = fr.read().split(',')

        # Makes set containing all objects once
        file_content = set(([str(i).replace('\n', '').strip() for i in file_content]))

        # Writes new '.txt'-file
        pathname, extension = filepath.split('.')
        new_file = pathname+"_ft"+'.'+extension
        with open(new_file, 'w') as fw:
            for i in file_content:
                fw.write(i + '\n')

        print(f"File reformatted into {new_file}!")

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        os.system(f"mv {filepath} {run_path}")
        print(f"Moved {filepath} to {run_path}!")
    else:
        print(f"File {filepath} skipped, has already been reformed!")

if __name__ == "__main__":
    files = glob("P110/*.txt")

    for i in sorted(files, key=lambda x: x[-8:]):
        sort_into_list(i)
