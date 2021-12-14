#!/usr/bin/env bash

#location of the data folders (target, calib, output/execution_directory)
TARGET=hd142666
FOLDER=nband
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/$TARGET/PRODUCTS/$FOLDER
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

do_oicalib mat_raw_estimates.2019-03-24T09_01_46.AQUARIUS.rb mat_raw_estimates.2019-03-24T08_48_04.AQUARIUS.rb
do_oicalib mat_raw_estimates.2019-03-24T09_01_46.AQUARIUS.rb mat_raw_estimates.2019-03-24T09_19_40.AQUARIUS.rb
do_oicalib mat_raw_estimates.2019-03-24T08_48_04.AQUARIUS.rb mat_raw_estimates.2019-03-24T09_19_40.AQUARIUS.rb

exit 0
