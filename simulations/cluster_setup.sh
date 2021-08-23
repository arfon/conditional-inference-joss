#!/bin/bash

if [[ $(hostname -s | grep "^hpcc[0-9]*$") ]]; then
    module load python/python-3.8.5
    source ../venv/bin/activate
fi

python cluster_setup.py "$@"
