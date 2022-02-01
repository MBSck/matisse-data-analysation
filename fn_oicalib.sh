#!/usr/bin/env bash
{
#location of the data folders (target, calib, output/execution_directory)
TARGET=suAur
FOLDER=nband
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/openTime/$TARGET/PRODUCTS/$FOLDER
EXECDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/scripts


do_oicalib() {
    RAWDIR=$DATADIR/$1
    CALIBDIR=$DATADIR/$2
    RESDIR=$DATADIR/calib_$FOLDER

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

do_oicalib mat_raw_estimates.2021-10-15T06_50_56.AQUARIUS.rb mat_raw_estimates.2021-10-15T07_20_19.AQUARIUS.rb

exit 0
}
