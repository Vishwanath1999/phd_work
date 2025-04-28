#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J rl_mrr_gru_sac
### -- ask for number of cores --
#BSUB -n 10

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### -- request 30GB of system memory --
#BSUB -R "rusage[mem=24GB]"
### -- set the email address --
#BSUB -u viswa@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o viswa_rl_mrr_gru_%J.out
#BSUB -e viswa_rl_mrr_gru_%J.err
# -- end of LSF options --

nvidia-smi

# Load the cuda module
module load cuda/12.4

# Activate virtual environment
source my_env/bin/activate

# Run your Python script within the virtual environment
python rl_mrr_cont_det_ctrl.py

# Deactivate virtual environment after execution (optional)
deactivate
