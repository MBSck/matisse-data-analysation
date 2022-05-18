"""The completely automated OB creation from parsing to upload"""

__author__ = "Marten Scheuck"
__date__   = "2022-05"

import sys
import getpass

from parseOBplan import parse_night_plan
from automatedOBcreation import ob_creation
from uploader import ob_uploader

def get_password():
    """Gets the user's ESO-user password to access the P2"""
    prompt = f'Input your ESO-user password: '
    if sys.platform == 'ios':
        import console
        password = console.password_alert(prompt)
    elif sys.stdin.isatty():
        password = getpass.getpass(prompt)
    else:
        password = input()

    return password

def main():
    # Path to the night observing plan in (.txt)-format
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/P109/"\
            "may2022/p109_observing_plan_v0.5.txt"

    # The output path
    outpath = "/Users/scheuck/Documents/PhD/matisse_stuff/observation/phase2/obs"

    # The resolution dict
    res_dict = {}

    # The period and the proposal tag of the run
    run_data = ["109", "2313"]

    # Get user password for ESO
    password = get_password()

    print("Parsing the Night plan!")
    print("-------------------------------------------------------------------")
    run_dict = parse_night_plan(path, save2file=True)
    print("Parsing complete!")
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")

    print("Creating the OBs!")
    print("-------------------------------------------------------------------")
    ob_creation("medium", outpath, run_data=run_dict,
                res_dict=res_dict, mode="gr")
    print("OB creation compete!")

    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
    print("Uploading the OBs!")
    print("-------------------------------------------------------------------")
    ob_uploader(p2, outpath, "production", run_data, "MbS", password)
    print("Uploading complete!")


if __name__ == "__main__":
    main()
