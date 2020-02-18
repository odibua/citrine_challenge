This repository implements an algorithm for sampling candidates from constrained high dimensional space.
This algorithm is detailed by S. Golchi et al in ["Monte Carlo based Designs for Constrained Domains"](https://arxiv.org/pdf/1512.07328.pdf). 

## System Requirements
To install and run this sampling method the following are required.
1. `pip3`
1. `python 3.6.*`

## Installation Instructions #
Below are steps to install this API:
1. `sudo pip3 install --upgrade pip setuptools wheel`
1. `git clone git@github.com:odibua/citrine_challenge.git && cd citrine_challenge`
1. `pip3 install -r requirements.txt`
1. `chmod +x sampler.sh`

## Run Instructions
1. `sampler.sh` assumes that python 3.6.* is run using `python3 <script>.py` If this is not the case
modify line 8 of `sampler.sh` to be `python sampler.py $1 $2 $3 $4`
1. `./sampler.sh <input_file> <output_file> <n_results>`