#!/bin/bash
INPUT_FILE=$1
OUTPUT_FILE=$2
N_RESULTS=$3
PLOT_BOOL=$4

echo $INPUT_FILE
python3 sampler.py $1 $2 $3 $4
