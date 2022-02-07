#!/usr/bin/env python3

__author__ = "Jacob Isbell"

from glob import glob
import numpy as np
from astropy.io import fits
from subprocess import call
from sys import argv
import sys

try:
    script, targdir, calibdir = argv
except:
    print("Usage: python3 do_oicalib.py /path/to/target/data/dir/ /path/to/calibrator/data/dir/")
    sys.exit(1)

targdir_fmat = targdir.split('/')[-1].split('mat_raw_estimates.')[-1]
calibdir_fmat = calibdir.split('/')[-1].split('mat_raw_estimates.')[-1]

outname = targdir_fmat + "_with_" + calibdir_fmat + "_CALIBRATED"
mysof = open(outname + '.sof', 'w'   )

try:
    print('Trying to make dir %s'%(outname))
    call('mkdir %s'%(outname), shell=True)

except:
    print('Directory %s exists, continuing...')


#for file in the targ dir, write to sof
try:
    targ_files = np.sort( glob(targdir + '/*RAW_INT_*.fits')  )
except:
    print("No TARGET_RAW_INT-files found -> Either calibrator or missing")
    targ_files = np.sort( glob(calibdir + '/*CALIB_RAW_INT_*.fits')  )

for tf in targ_files:
    print(tf)
    mysof.write("%s %s\n"%(tf, 'TARGET_RAW_INT') )

calib_files = np.sort( glob(calibdir + '/*CALIB_RAW_INT_*.fits')  )
if(len(calib_files) != len(targ_files)):
    print("N calib != N targ, trying to continue...")

for k in range( np.min([len(targ_files),len(calib_files)]) ):
    cf = calib_files[k]
    print(cf)
    mysof.write("%s %s\n"%(cf, 'CALIB_RAW_INT') )

mysof.close()

#okay, so the sof is written, now we can call esorex
call('esorex mat_cal_oifits %s'%(outname+'.sof'), shell=True)

#after calibration is done, move the files to the right folder
try:
    call('mv *CAL_INT_*.fits %s'%(outname), shell=True)
    call('mv esorex.log  %s'%(outname), shell=True)
except:
    print('Something went wrong, check the log file...')

