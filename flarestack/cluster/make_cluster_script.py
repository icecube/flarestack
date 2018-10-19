import os

from flarestack.shared import fs_dir, log_dir

username = os.path.basename(os.environ['HOME'])

root_dir = os.path.dirname(fs_dir[:-1])

cluster_dir = fs_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

submit_file = cluster_dir + "SubmitDESY.sh"

def make_submit_file():

    text = "#!/bin/zsh \n" \
           "## \n" \
           "##(otherwise the default shell would be used) \n" \
           "#$ -S /bin/zsh \n" \
           "## \n" \
           "##(the running time for this job) \n" \
           "#$ -l h_cpu=23:59:00 \n" \
           "#$ -l h_rss=6.0G \n" \
           "## \n" \
           "## \n" \
           "##(send mail on job's abort) \n" \
           "#$ -m a \n" \
           "## \n" \
           "##(stderr and stdout are merged together to stdout) \n" \
           "#$ -j y \n" \
           "## \n" \
           "## name of the job \n" \
           "## -N Flarestack script " + username + " \n" \
           "## \n" \
           "##(redirect output to:) \n" \
           "#$ -o /dev/null \n" \
           "## \n" \
           "sleep $(( ( RANDOM % 60 )  + 1 )) \n" \
           'exec > "$TMPDIR"/${JOB_ID}_stdout.txt ' \
           '2>"$TMPDIR"/${JOB_ID}_stderr.txt \n' \
           'eval $(/cvmfs/icecube.opensciencegrid.org/py2-v2/setup.sh) \n' \
           'export PYTHONPATH=' + root_dir + "/ \n" \
           '$SROOT/metaprojects/offline-software/V16-10-00/env-shell.sh ' \
           'python ' + fs_dir + 'core/minimisation.py -f $1 \n' \
           'cp $TMPDIR/${JOB_ID}_stdout.txt ' + log_dir + '\n'\
           'cp $TMPDIR/${JOB_ID}_stderr.txt ' + log_dir + '\n '

    print "No bash submit file found for DESY cluster!"
    print "Creating file at", submit_file

    with open(submit_file, "w") as f:
        f.write(text)

    print "Bash file created: \n"
    print text


if not os.path.exists(submit_file):
    make_submit_file()
