__author__ = "Marten Scheuck"

import MATISSE_create_OB_2 as ob

def make_science_obs(cal_lst, sci_lst,interferometric_array,  outdir):
    """Calls the 'ob.mat_gen_ob' with the right parameters that then creates the calibrations '.obx'-file

    Parameters
    ----------
    cal_lst: list
            The list with the calibration objects, the corresponding science objects need to be at the same index
        sci_lst: list
            The list with the science objects, the corresponding calibration objects need t be at the same index
        interferometric_array: str
            The array structure. Either 'large', 'medium' or 'small'
        outdir: str
            The output directory in which the '.obx'-file gets created
    """
    for i in sci_lst:
        ob.mat_gen_ob(i, interferometric_array,'SCI',outdir=outdir,obs_tpls=[ob.obs_ft_tpl],acq_tpl=ob.acq_ft_tpl)

def make_calibration_obs(cal_lst, sci_lst, tag_lst, interferometric_array, outdir):
    """Calls the 'ob.mat_gen_ob" with the right parameters that then creates the calibrations '.obx'-file
    
    Parameters
    ----------
    cal_lst: list
        The list with the calibration objects, the corresponding science objects need to be at the same index
    sci_lst: list
        The list with the science objects, the corresponding calibration objects need t be at the same index
    interferometric_array: str
        The array structure. Either 'large', 'medium' or 'small'
    outdir: str
        The output directory in which the '.obx'-file gets created
    lband: bool
        Determines if the calibration object is configured for L-band
    nband: bool
        Determines if the calibration object is configured for N-band
    """ 
    for i, o in enumerate(cal_lst):
        if isinstance(o, list):
            for j, l in enumerate(o):
                ob.mat_gen_ob(l,interferometric_array,'CAL',outdir=outdir,obs_tpls=[ob.obs_ft_tpl],acq_tpl=ob.acq_ft_tpl,sci_name=sci_lst[i],tag=tag_lst[i][j])
        else:
            ob.mat_gen_ob(o,interferometric_array,'CAL',outdir=outdir,obs_tpls=[ob.obs_ft_tpl],acq_tpl=ob.acq_ft_tpl,sci_name=sci_lst[i],tag=tag_lst[i])


if __name__ == "__main__":
    # Observation in the night of the 24th November 
    # sci_lst_003 = ["CQ_Tau", "DG_Tau", "beta_Pic", "R_Mon"]
    # cal_lst_003 = ["HD27482", "HD27482", "HD33042", ["HD49161", "HD58972"]]
    # tag_lst_003 = ["LN", "LN", "LN", ["LN", "N"]]
    # make_calibration_obs(cal_lst_003, sci_lst_003, tag_lst_003, "medium", outdir="/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.003_medium")
  
    # TODO: Check why the for 'Orion_BN' 'HD39853' is not found, same for 'HD38120' 'HD39853' -> Find way to skip these not found terms and display them afterwards
    # Observation in the night of the 22th/23th November early morning
    # sci_lst_004 = ["V892_Tau", "AB_Aur", "VY_Mon", "DG_Tau", "T_Tau_S", "Orion_BN", "HD259431", "HD31648"]
    # cal_lst_004 = [["HD17361", "HD20644"], "HD26526", "HD48433", "HD27482", "HD20893",["HD36167"], ["HD53510", "HD47886"], "HD43039"]
    # tag_lst_004 = [["LN", "N"], "LN", "LN", "LN", "LN", ["L", "LN"], ["LN", "LN"], "LN", "LN"]
    # make_calibration_obs(cal_lst_004, sci_lst_004, tag_lst_004, "medium", outdir="/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.004_medium")
   
    # Observation in the night of the 27th November
    # sci_lst_005 = ["HD38120", "VY_Mon", "HD259431", "R_Mon", "Orion_BN"]
    # cal_lst_005 = ["HD50778", ["HD58972", "HD61772"],["HD58972", "HD61421"], "HD58972", "HD58972"]
    # tag_lst_005 = ["LN", ["LN", "LN"], ["LN", "LN"], "LN", "LN"]
    # make_calibration_obs(cal_lst_005, sci_lst_005, tag_lst_005, "small", outdir="/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.005_small")
    
    """
    # This works!!!
    sci_lst_005 = ["VY_Mon", "HD38120"]
    cal_lst_005 = [["HD58972", "HD61772"], "HD58972"]
    outdir = '/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/obmaking/108.225V.005_small'
    for i, o in enumerate(cal_lst_005):
        if isinstance(o, list):
            for l in o:
                ob.mat_gen_ob(l,'medium','CAL',outdir=outdir, obs_tpls=[ob.obs_ft_tpl],acq_tpl=ob.acq_ft_tpl,sci_name=sci_lst_005[i],tag='LN')
        else:
            print(o)
            ob.mat_gen_ob(o,'medium','CAL',outdir=outdir, obs_tpls=[ob.obs_ft_tpl],acq_tpl=ob.acq_ft_tpl,sci_name=sci_lst_005[i],tag='LN')
    """

