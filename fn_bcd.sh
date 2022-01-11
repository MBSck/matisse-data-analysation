#!/usr/bin/env bash

#location of the data folders (target, calib, output/execution_directory)
TARGET=hd142666
FOLDER=calib_nband/UTs
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/$TARGET/PRODUCTS/$FOLDER
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts


do_bcd() {
    python3 bcd_calibration.py $1 &&
    mv -f ../plots/ $1
}

for dir in $DATADIR/*/
do
    echo "Start BCD of ${dir}"
    do_bcd $dir
done

exit 0
