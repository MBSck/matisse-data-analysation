#!/usr/bin/env bash
{
#location of the data folders
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/openTime/uxOri/PRODUCTS/nband/calib
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
}
