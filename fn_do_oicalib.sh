#!/usr/bin/env bash

#location of the data folders (target, calib, output)
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts
TARDIR=$DATADIR/lband/mat_raw_estimates.2019-05-14T05_28_03.HAWAII-2RG.rb

CALIBDIR=$DATADIR/lband/mat_raw_estimates.2019-05-14T06_12_59.HAWAII-2RG.rb

RESDIR=$DATADIR/calib

cd $EXECDIR

python3 do_oicalib.py $TARDIR $CALIBDIR &&

cd *.rb_CALIBRATED
mkdir sofAndMore &&

mv -f $EXECDIR/*.fits $EXECDIR/*.sof sofAndMore &&
cd $EXECDIR
mv -f *.rb_CALIBRATED $RESDIR
