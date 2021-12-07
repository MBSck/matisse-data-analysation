#!/bin/bash bash

#change this to point to your oca pipeline
cd ./oca_pipeline/tools/


#----------define the folders for saving data-----------------
#location of the data folders (raw, outputs, etc)
DATADIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/
CALIBDIR=/data/beegfs/astro-storage/groups/matisse/scheuck/data/hd142666/RAW/20190506
RAWDIR=$DATADIR/RAW/20190506
RESDIR=$DATADIR/PRODUCTS/

#name this the same as FOLDER to do lband reduction
CHECK_LBAND=false

#defaults to reducing lband data
start=`date +%s`

#----------run the pipeline-------------------------------
if [ "$CHECK_LBAND" == true ]
then
	python automaticPipeline.py --dirRaw=$RAWDIR --dirCalib=$CALIBDIR --dirResult=$RESDIR --nbCore=10 --overwrite=TRUE --maxIter=1 --paramL=/corrFlux=FALSE/coherentAlgo=2/compensate=[pb,rb,nl,if,bp,od]/cumulBlock=FALSE/spectralBinning=11/  --skipN  #--tplSTART=2021-02-28T08:13:00

	#------ move the results to a labelled folder ----------
    if [-d $RESDIR/lband]; then
	    mv -f $RESDIR/Iter1/*.rb $RESDIR/lband/
    else
        mkdir $RESDIR/lband/
	    mv -f $RESDIR/Iter1/*.rb $RESDIR/lband/
    fi
else
	python automaticPipeline.py --dirRaw=$RAWDIR --dirCalib=$CALIBDIR --dirResult=$RESDIR --nbCore=4 --overwrite=TRUE --maxIter=1 --paramN=/useOpdMod=TRUE/coherentAlgo=2/corrFlux=FALSE/compensate=[pb,rb,nl,if,bp,od]/coherentIntegTime=0.2/cumulBlock=FALSE/spectralBinning=11/replaceTel=2  --skipL 

	#------ move the results to a labelled folder ----------
    if [-d $RESDIR/nband]; then
	    mv -f $RESDIR/Iter1/*.rb $RESDIR/nband/
    else
        mkdir $RESDIR/nband/
	    mv -f $RESDIR/Iter1/*.rb $RESDIR/nband/
    fi
fi


end=`date +%s`
runtime=$((end-start))
printf "Finished calibration in %f sec" $runtime



exit 0

