#!/usr/bin/env bash
{
#location of the data folders (target, calib, output/execution_directory)
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/openTime/suAur/PRODUCTS/nband/calib
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts/ppdmodler/src/functionality

YLOWERVIS2=0.
YUPPERVIS2=0.6

do_plot() {
    cd $EXECDIR
    python3 plotter.py $1 $YLOWERVIS2 $YUPPERVIS2
}

for dir in $DATADIR/*/
do
    echo "Start plotting of folder ${dir}"
    do_plot $dir
done

exit 0
}
