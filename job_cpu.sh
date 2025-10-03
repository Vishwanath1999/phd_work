#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J Microcomb_Optimization
### -- ask for number of cores (match GA processes) --
#BSUB -n 20
### -- specify that cores must be on same host --
#BSUB -R "span[hosts=1]"
### -- memory per core (adjusted for PyTorch/GA) --
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
### -- walltime limit --
#BSUB -W 72:00
### -- notification settings --
#BSUB -B
#BSUB -N
### -- output files --
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err


### Activate virtual environment (if applicable)
source fdtd_py/bin/activate

### Run optimization
python torch_GA_test.py

# deactivate virtual environment (if applicable)
deactivate