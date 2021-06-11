#!/bin/bash - 
#======================================================
#
#          FILE: post_wrf.sh
# 
USAGE="./post_wrf.sh"
# 
#   DESCRIPTION: this code works in local_data/domain/d4(d3)/
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 05/11/2021 10:37
#      REVISION: 1.0
#=====================================================
set -o nounset           # Treat unset variables as an error
. ~/Shell/functions.sh   # ctang's functions

while getopts ":tf:" opt; do
    case $opt in
        t) TEST=1 ;;
        f) file=$OPTARG;;
        \?) echo $USAGE && exit 1
    esac
done
shift $(($OPTIND - 1))
#=================================================== 
domain=d4

function change_name_d3
{
   mv -v addout_d01_2015-12-25_00:00:00.nc addout_9km_${domain}.nc
   mv -v addout_d02_2015-12-25_00:00:00.nc addout_3km_${domain}.nc
   mv -v addout_d03_2015-12-25_00:00:00.nc addout_1km_${domain}.nc
}

function change_name_d4
{
   mv -v addout_d01_2015-12-25_00:00:00.nc addout_27km_${domain}.nc
   mv -v addout_d02_2015-12-25_00:00:00.nc addout_9km_${domain}.nc
   mv -v addout_d03_2015-12-25_00:00:00.nc addout_3km_${domain}.nc
   mv -v addout_d04_2015-12-25_00:00:00.nc addout_1km_${domain}.nc
}

function change_time
{
    for f in add*km*nc
    do 
        echo $f
        ncatted -O -a units,XTIME,m,c,'minutes since 2015-12-25 00:00:00' $f $f.temp
    done

    #checked=1
    #if [ checked ]
    #then
        #echo 'checked'
        #mv -v $f.temp $f
    #fi
}


function lastday
{
    for f in add*km*nc.temp
    do
        # select last day
        echo $f
        cdo seltimestep,97/113 $f $f.lastday.tmp
        cdo shifttime,4hours $f.lastday.tmp $f.lastday.localtime.temp
        rm *tmp
    done
}

function selvar
{
    for v in SWDOWN U V T2 Q2 W
    do
        for f in addout*lastday.localtime.temp
        do
            echo $v 
            cdo selvar,$v $f $v.${f%.temp}.nc
        done
    done
}


function rename
{
    for var in Q2 SWDOWN T2 W 
    do
        for f in $var.addout*temp*lastday.localtime.nc
        do
            echo $f
            ncrename -d west_east,lon -d south_north,lat $f
        done
    done

    for f in U.addout*temp*lastday.localtime.nc
    do
        echo $f
        ncrename -d west_east_stag,lon -d south_north,lat $f
    done

    for f in V.addout*temp*lastday.localtime.nc
    do
        echo $f
        ncrename -d west_east,lon -d south_north_stag,lat $f
    done
}
#-----------------------------

#-----------------------------
if [ $domain = "d3" ]
then
    echo 'd3'
    change_name_d3
else
    echo 'd4'
    change_name_d4
fi
#-----------------------------

rm *temp
change_time
lastday
selvar
rename
