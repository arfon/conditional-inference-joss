#!/bin/bash
#$ -j y
#$ -N sim
#$ -o output
#$ -q short.q

# run with:
# qsub -t 1-$(wc < -l variables.txt) simulation.sh

if [[ $(hostname -s | grep "^hpcc[0-9]*$") ]]; then
    module load python/python-3.8.5
    source ../venv/bin/activate
fi

VARS=$(sed -n ${SGE_TASK_ID}p variables.txt)
cd `cat root.txt`
python simulation.py $VARS
