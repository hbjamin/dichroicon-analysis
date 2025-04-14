#!/bin/bash
#SBATCH --job-name=mc_aggregate
#SBATCH --output=/nfs/disk1/users/jierans/junk/mc_aggregate_%A_%a.out
#SBATCH --array=0-75
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jierans@sas.upenn.edu

# Print the job ID and task ID
echo "This is host $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

python ./aggregate_mc_histogram.py ${SLURM_ARRAY_TASK_ID}
