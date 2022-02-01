#!/bin/env bash
# Brackets around the code makes it possible to change during running
# as whole file is parsed before run

# TODO: Make this work for different folders and subfolders, iteration and check results

{
#change this to point to your oca pipeline
cd ./oca_pipeline/tools/

#----------define the folders for saving data-----------------
# Location of the data folders (raw, outputs, etc)
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data

# Leave empty if only one folder is to be reduced
TARGETLIST=

# And give specific folder if only folder is to be reduced
TARGET=openTime/uxOri
TARGETDIR=$DATADIR/$TARGET

# Will define the folders if TARGETLIST is left empty
RAWDIR=$TARGETDIR/RAW
CALIBDIR=$RAWDIR
RESDIR=$TARGETDIR/PRODUCTS

# Checks if reductions for both bands should be done -> implement that
DO_BOTH=true

# Defaults to L-band data reduction 
CHECK_LBAND=false

make_directory() {
for i in "$@"
do
    if [ ! -d "$i" ]; then
        mkdir $i
    fi
done
}

#----------run the pipeline function-------------------------------
do_reduction() {
# $1: RAWDIR, $2: CALIBDIR, $3: RESDIR 

start=`date +%s`


#----------Do the L-band-------------------------------
if [ "$CHECK_LBAND" == true ]; then
	python automaticPipeline.py --dirRaw=$1 --dirCalib=$2 --dirResult=$3 --nbCore=10 --overwrite=TRUE --maxIter=1 --paramL=/corrFlux=FALSE/coherentAlgo=2/compensate=[pb,rb,nl,if,bp,od]/cumulBlock=FALSE/spectralBinning=11/  --skipN  #--tplSTART=2021-02-28T08:13:00

	#------ move the results to a labelled folder ----------
    if [ -d "$3/lband" ]; then
	    mv -f $3/Iter1/*.rb $3/lband
    else
        mkdir $3/lband/
	    mv -f $3/Iter1/*.rb $3/lband
    fi

#----------Do the N-band-------------------------------
else
	python automaticPipeline.py --dirRaw=$1 --dirCalib=$2 --dirResult=$3 --nbCore=4 --overwrite=TRUE --maxIter=1 --paramN=/useOpdMod=TRUE/coherentAlgo=2/corrFlux=FALSE/compensate=[pb,rb,nl,if,bp,od]/coherentIntegTime=0.2/cumulBlock=FALSE/spectralBinning=11/replaceTel=2  --skipL 

	#------ move the results to a labelled folder ----------
    if [ -d "$3/nband" ]; then
	    mv -f $3/Iter1/*.rb $3/nband
    else
        mkdir $3/nband/
	    mv -f $3/Iter1/*.rb $3/nband
    fi
fi

end=`date +%s`
runtime=$((end-start))
printf "Finished calibration in %f sec" $runtime
}

if [ -z $TARGETLIST];then
    # Make directories if they are not yet exisiting
    make_directory $TARGETDIR $CALIBDIR $RESDIR

    do_reduction $RAWDIR $CALIBDIR $RESDIR
else
    for i in $TARGETLIST
    do
        # Temp variables that set the folders
        TEMPRAW=$TARGETDIR/$i/RAW
        TEMPCALIB=$TEMPCALIB
        TEMPRES=$TARGETDIR/$i/PRODUCTS

        # Make directories if they are not yet exisiting
        make_directory $TEMPRAW $TEMPCALIB

        do_reduction $TEMPRAW $TEMPCALIB $TEMPRES
    done
fi

exit
}
