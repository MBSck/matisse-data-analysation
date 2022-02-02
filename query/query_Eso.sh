#!/usr/bin/env bash

user_name="MbS"
password="QekutafAmeNGNVZ"
instrument_name="matisse"
max_rows=1500

read -p "Target:" target
read -p "StartDate:" start_date
read -p "EndDate:" end_date
read -p "Download (Y/N):" download

/bin/bash /home/scheuck/MATISSE/programms/python/getdata_ESO_archive.py -User $user_name -Password $password -TargetName $target -Inst $instrument_name -StartDate $start_date -EndDate $end_date

