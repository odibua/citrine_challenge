#!/bin/bash
INPUT_FILE=$1
OUTPUT_FILE=$2
N_RESULTS=$3

echo $INPUT_FILE
python3 sampler.py $1 $2 $3