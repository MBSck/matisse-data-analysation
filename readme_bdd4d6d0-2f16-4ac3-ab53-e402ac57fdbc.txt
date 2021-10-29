This file contains an overview of the structure and content of your download request from the ESO Science Archive.


For every downloaded dataset, files are listed below with the following structure:

dataset_name
        - archive_file_name (technical name, as saved on disk)	original_file_name (user name, contains relevant information) category size


Please note that, depending on your operative system and method of download, at download time the colons (:) in the archive_file_name as listed below may be replaced by underscores (_).


In order to rename the files on disk from the technical archive_file_name to the the more meaningful original_file_name, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print "test -f",$2,"&& mv",$2,$3}' | sh


In case you have requested cutouts, the file name on disk contains the TARGET name that you have provided as input. To order files by it when listing them, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print $2}' | sort -t_ -k3,3


Publications based on observations collected at ESO telescopes must acknowledge this fact (please see: http://archive.eso.org/cms/eso-data-access-policy.html#acknowledgement).

Your feedback regarding the data quality of the downloaded data products is greatly appreciated by contacting the ESO Archive Science Group via https://support.eso.org/ , subject: Phase 3 ... thanks!!

The downloaded data are characterized in detail in the following release documents:


MATIS.2019-08-28T23:28:25.205 
	- MATIS.2019-08-28T23:28:25.205		Raw data for which no processed data is available	46205
