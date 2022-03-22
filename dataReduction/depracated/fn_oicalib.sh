#!/usr/bin/env bash
{
#location of the data folders (target, calib, output/execution_directory)
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/openTime/pds73/PRODUCTS/nband
EXECDIR=/home/scheuck/scripts


do_oicalib() {
    RAWDIR=$DATADIR/$1
    CALIBDIR=$DATADIR/$2
    RESDIR=$DATADIR/calib

    if ! [ -d "$RESDIR" ]; then
        mkdir $RESDIR
    fi

    cd $EXECDIR
    python3 do_oicalib.py $RAWDIR $CALIBDIR &&

    cd *.rb_CALIBRATED
    mkdir sofAndMore &&

    mv -f $EXECDIR/*.fits $EXECDIR/*.sof sofAndMore &&
    cd $EXECDIR
    mv -f *.rb_CALIBRATED $RESDIR

}

# for i in "${!RAWLIST[@]}"
# do
#    do_oicalib "${RAWLIST[i]}" "${CALIBLIST[i]}"
# done


# for i in "${!RAWLIST[@]}"
# do
#     printf "%s is in %s\n" "$i" "${RAWLIST[$i]}"
# done
do_oicalib mat_raw_estimates.2022-02-19T06_00_08.AQUARIUS.rb mat_raw_estimates.2022-02-19T06_31_49.AQUARIUS.rb
do_oicalib mat_raw_estimates.2022-02-19T07_00_22.AQUARIUS.rb mat_raw_estimates.2022-02-19T06_31_49.AQUARIUS.rb
do_oicalib mat_raw_estimates.2022-02-19T06_00_08.AQUARIUS.rb mat_raw_estimates.2022-02-19T07_00_22.AQUARIUS.rb

exit 0
}
