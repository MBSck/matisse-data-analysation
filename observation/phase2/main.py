"""The completely automated OB creation from parsing to upload"""

__author__ = "Marten Scheuck"
__date__   = "2022-05"

from parseOBplan import parse_night_plan
from automatedOBcreation import ob_creation
from uploader import ob_uploader

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

    run_dict = parse_night_plan(path, save2file=False)
    ob_creation("medium", outpath, run_data=run_dict, res_dict=res_dict, mode="gr")
    ob_uploader(outpath, "production", run_data, "MbS")


if __name__ == "__main__":
    main()
