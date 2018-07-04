#!/bin/zsh
##
##(otherwise the default shell would be used)
#$ -S /bin/zsh
##
##(the running time for this job)

#$ -l h_cpu=23:59:00
#$ -l h_rss=4.5G
##
##
##(send mail on job's end and abort)
#$ -m a
##
##(stderr and stdout are merged together to stdout)
#$ -j y
##

## name of the job
## -N TDE Stacking Analysis
##
##(redirect output to:)
#$ -o /dev/null
##

sleep $(( ( RANDOM % 60 )  + 1 ))

exec > "$TMPDIR"/${JOB_ID}_stdout.txt 2>"$TMPDIR"/${JOB_ID}_stderr.txt

eval $(/cvmfs/icecube.opensciencegrid.org/py2-v2/setup.sh)

export PYTHONPATH=/afs/ifh.de/user/s/steinrob/flarestack/

$SROOT/metaprojects/offline-software/V16-10-00/env-shell.sh python /afs/ifh.de/user/s/steinrob/flarestack/core/minimisation.py -f $1

cp $TMPDIR/${JOB_ID}_stdout.txt /afs/ifh.de/user/s/steinrob/scratch/flarestack__data/logs
cp $TMPDIR/${JOB_ID}_stderr.txt /afs/ifh.de/user/s/steinrob/scratch/flarestack__data/logs
