#!/bin/bash

## Job name
#SBATCH --job-name=Exercice_2_Bootstrap

## Account to charge
#SBATCH --account=mnl03


## Specify the partition/queue
#SBATCH --partition=normal 
  
## Number of nodes 
#SBATCH --nodes=2 
  
## Number of tasks (cores) per node   
#SBATCH --ntasks=2 

## Number of CPUs per task  
#SBATCH --cpus-per-task=16  # Adjust this to the number of cores you need    
    
## Memory per node (in MB) 
#SBATCH --mem=16000 # Adjust memory according to your parallel requirements

## Maximum execution time
#SBATCH --time=0-24:00:00 


#SBATCH --output=mnl03_sui.out
#SBATCH --error=my_job_standard_output.out   

## Email notification 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mnl03@mail.aub.edu

## Load Python module (if necessary)
module load python/3  # Adjust this to your cluster's Python version

source /home/mnl03/virt_TENSORFLOW/bin/activate

## Execute the Python script (specify the number of cores via environment variable)
export JOBLIB_NUM_THREADS=$SLURM_CPUS_PER_TASK
python Exercise_2_many_n.py

## Deactivate the virtual environment
deactivate
