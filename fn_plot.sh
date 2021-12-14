#!/usr/bin/env bash

#location of the data folders (target, calib, output/execution_directory)
TARGET=hd142666
FOLDER=calib_nband
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/$TARGET/PRODUCTS/$FOLDER
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts

YLOWERVIS2=0.
YUPPERVIS2=10.

do_plot() {
    python3 myplotter.py $1 $YLOWERVIS2 $YUPPERVIS2
}

for dir in $DATADIR/*/
do
    echo "Start plotting of folder ${dir}"
    do_plot $dir
done

exit 0
