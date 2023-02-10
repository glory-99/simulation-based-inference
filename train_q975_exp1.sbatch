#!/bin/sh

#SBATCH --job-name=exp1
#SBATCH --account=pi-jingshuw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads
#SBATCH --output=./output/MyJob_%j.out
#SBATCH --error=./output/MyJob_%j.err

# LOAD MODULES
module load pytorch

# DO COMPUTE WORK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python training_general.py 50 0.975 --epochs 1000 --exp_id 1
