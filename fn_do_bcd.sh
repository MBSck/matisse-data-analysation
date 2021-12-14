#!/usr/bin/env bash

#location of the data folders (target, calib, output/execution_directory)
TARGET=hd142666
FOLDER=calib_nband
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/$TARGET/PRODUCTS/$FOLDER
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts


do_bcd() {
    python3 bcd_calibration.py $1
}

for dir in $DATADIR/*/
do
    echo "Start BCD of ${dir}"
    # do_bcd 
done

do_bcd /data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/PRODUCTS/calib_nband/2019-03-24T08_48_04.AQUARIUS.rb_with_2019-03-24T09_19_40.AQUARIUS.rb_CALIBRATED/

exit 0
