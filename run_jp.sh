#!binbash 
### General options 
### -- specify queue -- 
#BSUB -q fotonano
### -- set the job Name -- 
#BSUB -J jupyter_nb
### -- ask for number of cores (default 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R span[hosts=1]
### -- specify that we need 4GB of memory per coreslot -- 
#BSUB -R rusage[mem=32GB]
### -- specify that we want the job to get killed if it exceeds 5 GB per coreslot -- 
#BSUB -M 32GB
### -- set walltime limit hhmm -- 
#BSUB -W 1200
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o zhomef29180375logsoutnb_%J.out 
#BSUB -e zhomef29180375logserrnb_%J.err 

# Define the notebook port
NOTEBOOKPORT=8008

# Check if $LSB_SUB_HOST is not empty
if [ -n $LSB_SUB_HOST ]; then
    SSH_HOST=$LSB_SUB_HOST
else
    SSH_HOST=$PBS_O_HOST
fi

# Check if either variable is empty (not both)
if [ -z $SSH_HOST ]; then
    echo Error Both LSB_SUB_HOST and PBS_O_HOST are empty.
    exit 1
fi

# Function to clean up and close the SSH tunnel
cleanup() {
    echo Closing the SSH tunnel...
    pkill -f ssh -N -f -R $NOTEBOOKPORTlocalhost$NOTEBOOKPORT $SSH_HOST
    echo Tunnel closed.
}

# Trap signals and call cleanup when the script exits
trap cleanup EXIT

# Establish SSH tunnel based on chosen host
ssh -N -f -R $NOTEBOOKPORTlocalhost$NOTEBOOKPORT $SSH_HOST
if [ $ -ne 0 ]; then
    echo Failed to establish SSH tunnel.
    exit 1
fi


# Start the Jupyter notebook server
jupyter notebook --port=$NOTEBOOKPORT --no-browser --certfile=zhomeb21212145.jupyterjupy_ssl_cert.pem --keyfile=zhomeb21212145.jupyterjupy_ssl_key.key