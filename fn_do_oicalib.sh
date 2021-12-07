#!/usr/bin/env bash

#location of the data folders (target, calib, output/execution_directory)
TARGET=hd142666
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/$TARGET/PRODUCTS
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts

RAWLIST=()
CALIBLIST=()

do_oicalib() {
    FOLDER=nband
    RAWDIR=$DATADIR/$FOLDER
    CALIBDIR=$DATADIR/$FOLDER

    RESDIR=$DATADIR/calib

    cd $EXECDIR
    python3 do_oicalib.py $RAWDIR $CALIBDIR &&

    cd *.rb_CALIBRATED
    mkdir sofAndMore &&

    mv -f $EXECDIR/*.fits $EXECDIR/*.sof sofAndMore &&
    cd $EXECDIR
    mv -f *.rb_CALIBRATED $RESDIR

    exit 0
}

for i in "${!RAWDIR[@]}"
do
    do_oicalib "${}" "${}" &&
