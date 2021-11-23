__author__ = "Marten Scheuck"

import os
import MATISSE_create_OB_2 as ob

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

def make_sci_obs(sci_lst, interferometric_array_config, outdir) -> None:
    """Gets the inputs from a list and calls the 'mat_gen_ob' for every list element

    Parameters
    ----------
    sci_lst: list
        Contains the science objects
    interferometric_array_config: str
        The array configuration ('small', 'medium', 'large')
    outdir: str
        The output directory, where the '.obx'-files will be created in

    Returns
    -------
    None
    """
    for i in sci_lst:
        ob.mat_gen_ob(i, interferometric_array_config ,'SCI',outdir=outdir,
            obs_tpls=[ob.obs_ft_tpl],acq_tpl=ob.acq_ft_tpl)

def make_cal_obs(cal_lst, sci_lst, tag_lst, interferometric_array_config, outdir) -> None:
    """Checks if there are sublists in the calibration list and calls the 'mat_gen_ob' with the right inputs
    to generate the calibration objects.
    The input lists correspond to each other index-wise (e.g., cal_lst[1], sci_lst[1], tag_lst[1]; etc.)
    
    Parameters
    ----------
    cal_lst: list
        Contains the calibration objects corresponding to the science objects
    sci_lst: list
        Contains the science objects
    tag_lst: list
        Contains the tags (either 'L', 'N', or both) and corresponds to the science objects
    interferometric_array_config: str
        The array configuration ('small', 'medium', 'large')
    outdir: str
        The output directory, where the '.obx'-files will be created in
    
    Returns
    -------
    None
    """
    for i, o in enumerate(cal_lst):
        if isinstance(o, list):
            for j, l in enumerate(o):
                ob.mat_gen_ob(l, interferometric_array_config, 'CAL', outdir=outdir,
                    obs_tpls=[ob.obs_ft_tpl], acq_tpl=ob.acq_ft_tpl, sci_name=sci_lst[i], tag=tag_lst[i][j])
        else:
            ob.mat_gen_ob(o, interferometric_array_config, 'CAL', outdir=outdir,
                obs_tpls=[ob.obs_ft_tpl], acq_tpl=ob.acq_ft_tpl, sci_name=sci_lst[i],tag=tag_lst[i])

if __name__ == "__main__":
    # Run '.004'
    # sci_lst_004 = ["V892_Tau", "DG_Tau", "CQ_Tau", "VY_Mon", "R_Mon", "T_Tau_S", "T_Tau_N", "HD31648", "HD259431", "Orion_BN"]
    # cal_lst_004 = [["HD17361", "HD20644"], "HD27482", "HD27482", "HD37160", ["HD49161", "HD58972"], "HD20893", "HD20893", "HD27482", "HD37160", ["HD39400", "HD42042", "HD50778"]]
    # tag_lst_004 = [["LN", "N"], "LN", "LN", "LN", ["LN", "LN"], "LN", "LN", "LN", "LN", ["L", "N", "N"]]
    # make_cal_obs(cal_lst_004, sci_lst_004, tag_lst_004, "medium",  "/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.004_medium")

    # sci_lst_003 = ["AB_Aur", "beta_Pic", "HD38120", "R_Mon"]
    # cal_lst_003 = ["HD26526", "HD39523", "HD50778", ["HD49161", "HD58972"]]
    # tag_lst_003 = ["LN", "LN", "LN", ["LN", "LN"]]
    #make_cal_obs(cal_lst_003, sci_lst_003, tag_lst_003, "medium",  "/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.003_medium")

    # Make sci-file
    # make_sci_obs(["R_Scl", "HD72106", "HD87643", "HD98922"], "small", "/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking")

    # Make calibs for sci-file
    # sci_lst_backup = ["R_Scl", "HD87643", "HD72106", "HD98922"]
    # cal_lst_backup = [["HD6595", "HD9053"], "HD84810", "HD66435", "HD92436"]
    # tag_lst_backup = [["L", "N"], "LN", "LN", "LN"]
    # make_cal_obs(cal_lst_backup, sci_lst_backup, tag_lst_backup, "medium", "/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.backup_medium")

    # For shell implementation
    shell_main()

